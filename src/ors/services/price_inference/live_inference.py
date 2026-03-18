"""Live inference: generate next-24-hour electricity price predictions.

Based on API outputs (Weather + Price).

Supports two model types:

* **LightGBM ForecasterRecursive** (default) — saved as ``.joblib`` by the
  training pipeline.  Loaded from ``LGBM_MODEL_DIR`` (most recent file wins).
  Runs at native 15-min resolution; no resampling required.

* **XGBoost** (legacy) — saved as ``model.pkl``.  Loaded from ``MODEL_PATH``.
  Generates hourly predictions that are forward-filled to 15-min resolution.

To switch models, pass ``--model`` on the command line pointing to either a
``.joblib`` / ``.pkl`` file **or** a directory containing LGBM joblib files.

Run from the repo root::

    python -m ors.services.price_inference.live_inference
"""

from __future__ import annotations

import argparse
import pickle  # used for legacy XGBoost .pkl models
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ors.services.prediction.data_pipeline import build_merged_dataset, get_feature_cols
from ors.services.price_inference.live_data_pipeline import (
    LGBM_PRICE_LOOKBACK_HOURS,
    PRICE_LOOKBACK_HOURS,
    build_live_lgbm_dataset,
    build_live_merged_dataset,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Directory that holds the LGBM inference joblib files (written by training).
LGBM_MODEL_DIR: Path = Path("models/price_prediction/lgbm_recursive_single_model")

# Legacy XGBoost model path.
MODEL_PATH: Path = Path("models/price_prediction/model.pkl")

# Number of 15-min steps the LGBM model predicts in one shot (= HORIZON used
# during training, 96 steps = 24 h).
LGBM_HORIZON_STEPS: int = 96

# Number of hours ahead to predict (legacy XGBoost path).
HORIZON_HOURS: int = 24

# Lag steps — must match those used during XGBoost training.
LAG_STEPS: tuple[int, ...] = (1, 2, 3, 6, 12, 24)

# Path to write the 15-minute predictions CSV consumed by the optimizer.
OUTPUT_PATH: Path = Path("data/live_price_predictions.csv")


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------


def find_latest_lgbm_model(model_dir: Path) -> tuple[Path, Path | None]:
    """Return (model_path, meta_path) for the most recent LGBM joblib artifact.

    The training pipeline writes files named::

        recursive_single_model_<YYYYMMDD_HHMMSS>.joblib
        recursive_single_model_<YYYYMMDD_HHMMSS>_meta.joblib

    The most recently written non-meta file is selected.

    Args:
        model_dir: Directory to search.

    Returns:
        Tuple of the model file path and its companion meta file path (or
        ``None`` if the meta file does not exist).

    Raises:
        FileNotFoundError: If no model files are found in *model_dir*.
    """
    if not model_dir.exists():
        raise FileNotFoundError(
            f"LGBM model directory not found: {model_dir}\n"
            "Run the training script first:\n"
            "  python -m ors.services.prediction.train_script --model-name lgbm_recursive"
        )
    candidates = sorted(
        f for f in model_dir.glob("recursive_single_model_*.joblib") if not f.stem.endswith("_meta")
    )
    if not candidates:
        raise FileNotFoundError(
            f"No LGBM model files found in {model_dir}.\n"
            "Expected pattern: recursive_single_model_<timestamp>.joblib"
        )
    model_path = candidates[-1]
    meta_path = model_dir / (model_path.stem + "_meta.joblib")
    return model_path, (meta_path if meta_path.exists() else None)


def load_model(model_path: Path) -> Any:
    """Load a model from disk.

    Supports both ``.joblib`` (LGBM / skforecast) and ``.pkl`` (XGBoost)
    formats.  When *model_path* is a **directory**, the most recently written
    LGBM joblib file inside it is loaded automatically.

    Args:
        model_path: Path to a ``.joblib`` / ``.pkl`` model file, or a
            directory containing LGBM joblib artifacts.

    Returns:
        Deserialised model object.

    Raises:
        FileNotFoundError: If the file (or directory) does not exist.
    """
    if model_path.is_dir():
        model_path, _ = find_latest_lgbm_model(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Update MODEL_PATH / LGBM_MODEL_DIR or pass --model on the CLI."
        )

    if model_path.suffix == ".joblib":
        import joblib

        model = joblib.load(model_path)
    else:
        with open(model_path, "rb") as fh:
            model = pickle.load(fh)

    print(f"Model loaded: {model_path}  ({type(model).__name__})")
    return model


def _load_lgbm_meta(model_path: Path) -> dict | None:
    """Try to load the companion ``_meta.joblib`` file for a LGBM model.

    Args:
        model_path: Path to the model ``.joblib`` file.

    Returns:
        Meta dictionary, or ``None`` if the file is not found.
    """
    import joblib

    meta_path = model_path.parent / (model_path.stem + "_meta.joblib")
    if meta_path.exists():
        return joblib.load(meta_path)  # type: ignore[no-any-return]
    return None


def _is_lgbm_forecaster(model: Any) -> bool:
    """Return ``True`` if *model* is a skforecast ``ForecasterRecursive``."""
    return type(model).__name__ == "ForecasterRecursive"


# ---------------------------------------------------------------------------
# Feature preparation (XGBoost / legacy path)
# ---------------------------------------------------------------------------


def prepare_features_for_inference(
    df: pd.DataFrame,
    model: Any,
    target_col: str = "Price",
) -> pd.DataFrame:
    """Extract a feature matrix from df aligned exactly to the model's expected columns.

    Uses model.feature_names_in_ to select and reorder columns to precisely match
    what the model was trained on, avoiding any feature name or order mismatch.
    Boolean columns are cast to int. Any missing columns are filled with 0.
    Any extra columns not expected by the model are silently dropped.

    Args:
        df: Merged DataFrame.
        model: Loaded model with a feature_names_in_ attribute.
        target_col: Name of the target column to drop if present.

    Returns:
        Feature DataFrame with exactly the columns the model expects, in the
        correct order, ready for model.predict().
    """
    features = df.copy()

    # Drop target and Timestamp if present
    for candidate in [target_col, "price", "target_price", "Price"]:
        if candidate in features.columns:
            features = features.drop(columns=[candidate])
            break

    if "Timestamp" in features.columns:
        features = features.drop(columns=["Timestamp"])

    # Cast booleans to int so they are numeric
    for col in features.columns:
        if features[col].dtype == "bool":
            features[col] = features[col].astype(int)

    # Fill NaNs with column medians before reindexing
    for col in features.columns:
        if features[col].isna().any():
            features[col] = features[col].fillna(features[col].median())

    # Align to exactly the columns the model expects, in the correct order.
    # Missing columns are filled with 0; extra columns are dropped.
    expected_cols: list[str] = list(model.feature_names_in_)
    features = features.reindex(columns=expected_cols, fill_value=0)

    # Ensure every column is numeric — reindex can introduce object dtype
    # for columns that were bool before reindexing (e.g. IsWeekend, IsHoliday).
    features = features.apply(pd.to_numeric, errors="coerce").fillna(0)

    return features


# ---------------------------------------------------------------------------
# Forecast row selection (XGBoost / legacy path)
# ---------------------------------------------------------------------------


def select_forecast_rows(
    live_df: pd.DataFrame,
    horizon_hours: int,
    reference_time: datetime | None = None,
) -> pd.DataFrame:
    """Return the next horizon_hours rows starting from the reference hour.

    Filters to rows whose Timestamp is at or after reference_time (or now in
    live mode), then takes the first horizon_hours of those.

    Args:
        live_df: Full merged DataFrame sorted by Timestamp.
        horizon_hours: Number of forecast rows to return.
        reference_time: If provided, use this as the start of the forecast
            window instead of the current time.

    Returns:
        First horizon_hours rows at or after the reference hour.
    """
    if reference_time is not None:
        ts = pd.Timestamp(reference_time)
        now = (ts.tz_convert(None) if ts.tzinfo is not None else ts).floor("h")
    else:
        now = pd.Timestamp.utcnow().tz_convert(None).floor("h")

    future_df = live_df[live_df["Timestamp"] >= now]
    forecast_df = future_df.head(horizon_hours).copy()

    if forecast_df.empty:
        print("Warning: no forecast rows found at or after the current hour.")

    return forecast_df


# ---------------------------------------------------------------------------
# LGBM ForecasterRecursive inference helpers
# ---------------------------------------------------------------------------


def _extract_lgbm_inputs(
    df: pd.DataFrame,
    feature_cols: list[str],
    window_size: int,
    forecast_steps: int,
    reference_time: datetime | None = None,
) -> tuple[pd.Series, pd.DataFrame, pd.DatetimeIndex]:
    """Extract last_window, exog, and forecast timestamps from a merged DataFrame.

    Works for both live datasets (future rows have ``price = NaN``) and CSV
    datasets (all rows have prices — forward looking rows are selected by
    timestamp relative to *reference_time*).

    Args:
        df: DataFrame with a ``Timestamp`` and ``price`` column, as returned by
            :func:`~live_data_pipeline.build_live_lgbm_dataset` or
            :func:`~ors.services.prediction.data_pipeline.build_merged_dataset`.
        feature_cols: Ordered list of exogenous feature column names.
        window_size: Number of lagged price steps the forecaster needs
            (``model.window_size``).
        forecast_steps: Number of 15-min steps to predict.
        reference_time: UTC datetime treated as "now".  If ``None``, uses the
            current time.

    Returns:
        Tuple of:
        * ``last_window`` — ``pd.Series`` of the last *window_size* observed
          prices with a clean integer index.
        * ``exog`` — ``pd.DataFrame`` of exog features for the next
          *forecast_steps* rows with a clean integer index.
        * ``forecast_index`` — ``pd.DatetimeIndex`` of 15-min timestamps for
          the forecast window.
    """
    if reference_time is not None:
        _ts = pd.Timestamp(reference_time)
        now = (_ts.tz_convert(None) if _ts.tzinfo is not None else _ts).floor("15min")
    else:
        now = pd.Timestamp.utcnow().tz_convert(None).floor("15min")

    # All rows before "now" at 15-min resolution feed the lag window.
    # The live price API returns half-hourly data, so after reindexing to 15-min
    # the intermediate slots have NaN price.  Forward-fill to propagate the
    # last known price into those gaps before building the window.
    past_df = df[df["Timestamp"] < now].copy()
    past_df["price"] = past_df["price"].ffill()

    non_null_count = past_df["price"].notna().sum()
    if non_null_count == 0:
        print(
            f"Warning: no price observations available in the past window; "
            f"LGBM requires {window_size}.  Predictions may be inaccurate."
        )
    elif len(past_df) < window_size:
        print(
            f"Warning: only {len(past_df)} past rows available; "
            f"LGBM requires {window_size}.  Predictions may be less accurate."
        )
    last_window_series = past_df["price"].iloc[-window_size:].reset_index(drop=True)

    # Future exog: rows at or after "now" (price may be NaN for live data,
    # or real for CSV/backtest mode — either way we only use feature_cols).
    future_df = df[df["Timestamp"] >= now].head(forecast_steps).copy()
    actual_steps = len(future_df)

    if actual_steps < forecast_steps:
        print(
            f"Warning: only {actual_steps} future exog rows available; "
            f"need {forecast_steps}.  Truncating forecast horizon."
        )

    missing_cols = [c for c in feature_cols if c not in future_df.columns]
    if missing_cols:
        print(f"Warning: missing exog columns (will be filled with 0): {missing_cols}")

    exog_df = future_df.reindex(columns=feature_cols, fill_value=0.0).reset_index(drop=True)
    # skforecast requires exog index to start one step after last_window ends.
    # last_window has index 0..window_size-1, so exog must start at window_size.
    exog_df.index = exog_df.index + window_size

    # Build forecast DatetimeIndex
    forecast_index = pd.date_range(start=now, periods=actual_steps, freq="15min")

    return last_window_series, exog_df, forecast_index


# ---------------------------------------------------------------------------
# Full pipeline (importable entry point for next modules)
# ---------------------------------------------------------------------------


def run_inference(
    model_path: Path | None = None,
    horizon_hours: int = HORIZON_HOURS,
    lag_steps: tuple[int, ...] = LAG_STEPS,
    output_path: Path = OUTPUT_PATH,
    reference_time: datetime | None = None,
    use_csv: bool = False,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """Run the full price inference pipeline.

    Automatically detects the model type:

    * If *model_path* points to (or resolves to) a ``.joblib`` file, a
      ``ForecasterRecursive`` LGBM model is assumed.  Data is fetched via
      :func:`~live_data_pipeline.build_live_lgbm_dataset` (or
      :func:`~ors.services.prediction.data_pipeline.build_merged_dataset`
      when ``use_csv=True``) and predictions are produced at native 15-min
      resolution.

    * Otherwise a legacy XGBoost ``.pkl`` model is assumed.  The existing
      hourly pipeline runs unchanged.

    Args:
        model_path: Path to the model file or directory.  Defaults to
            ``LGBM_MODEL_DIR`` (LGBM) with fallback to ``MODEL_PATH``
            (XGBoost).
        horizon_hours: Hours ahead to predict (XGBoost path) or used to
            derive ``forecast_steps`` for the LGBM path (``horizon_hours * 4``
            steps at 15-min resolution).
        lag_steps: Lag offsets in hours — XGBoost path only.
        output_path: Path to write the predictions CSV.
        reference_time: UTC datetime to treat as "now".  If None, runs live.
        use_csv: If True, build features from the training CSVs in Data/
            instead of calling live APIs.
        project_root: Repo root used when use_csv=True.  Defaults to four
            directories above this file.

    Returns:
        DataFrame with columns ``Timestamp`` and ``Price_pred``
        at 15-minute resolution.
    """
    root = project_root or Path(__file__).resolve().parents[4]

    # ------------------------------------------------------------------
    # Resolve model path / directory
    # ------------------------------------------------------------------
    resolved_model_path: Path
    if model_path is None:
        # Auto-detect: prefer LGBM if directory exists and contains models
        try:
            resolved_model_path, _ = find_latest_lgbm_model(LGBM_MODEL_DIR)
        except FileNotFoundError:
            resolved_model_path = MODEL_PATH
    elif model_path.is_dir():
        resolved_model_path, _ = find_latest_lgbm_model(model_path)
    else:
        resolved_model_path = model_path

    # ------------------------------------------------------------------
    # Step 2 — Load model (done early so we can branch on model type)
    # ------------------------------------------------------------------
    print("\n-- Step 2: Loading model")
    model = load_model(resolved_model_path)

    is_lgbm = _is_lgbm_forecaster(model)

    # ------------------------------------------------------------------
    # Step 1 — Build dataset
    # ------------------------------------------------------------------
    if use_csv:
        mode = f"CSV ({reference_time})" if reference_time is not None else "CSV (all)"
        print(f"\n-- Step 1: Building {mode} merged dataset from training CSVs")
        live_df = build_merged_dataset(root)
    elif is_lgbm:
        mode = f"historical ({reference_time})" if reference_time is not None else "live"
        forecast_steps = horizon_hours * 4
        print(f"\n-- Step 1: Building {mode} LGBM merged dataset")
        live_df = build_live_lgbm_dataset(
            forecast_steps=forecast_steps,
            past_hours=LGBM_PRICE_LOOKBACK_HOURS,
            reference_time=reference_time,
        )
    else:
        mode = f"historical ({reference_time})" if reference_time is not None else "live"
        print(f"\n-- Step 1: Building {mode} merged dataset")
        live_df = build_live_merged_dataset(
            past_hours=PRICE_LOOKBACK_HOURS,
            forecast_days=3,
            lag_steps=lag_steps,
            reference_time=reference_time,
        )
    print(f"Dataset shape: {live_df.shape}")

    # ------------------------------------------------------------------
    # LGBM ForecasterRecursive path
    # ------------------------------------------------------------------
    if is_lgbm:
        forecast_steps = horizon_hours * 4

        # Load feature_cols from meta (preferred) or derive from DataFrame
        meta = _load_lgbm_meta(resolved_model_path)
        if meta is not None and "feature_cols" in meta:
            feature_cols: list[str] = meta["feature_cols"]
            print(f"Feature cols loaded from meta: {len(feature_cols)} columns")
        else:
            feature_cols = get_feature_cols(live_df)
            print(f"Feature cols derived from dataset: {len(feature_cols)} columns")

        window_size = int(getattr(model, "window_size", 672))

        print(
            f"\n-- Step 3: Extracting last_window ({window_size} steps) and exog ({forecast_steps} steps)"
        )
        last_window, exog, forecast_index = _extract_lgbm_inputs(
            live_df,
            feature_cols=feature_cols,
            window_size=window_size,
            forecast_steps=forecast_steps,
            reference_time=reference_time,
        )
        actual_steps = len(exog)

        print(f"\n-- Step 4: Running LGBM inference ({actual_steps} steps)")
        pred_series = model.predict(
            steps=actual_steps,
            last_window=last_window,
            exog=exog,
        )

        results = pd.DataFrame(
            {
                "Timestamp": forecast_index[:actual_steps],
                "Price_pred": pred_series.to_numpy(),
            }
        )
        print(f"15-minute rows: {len(results)}  (expected {forecast_steps})")

        print("\n-- Predictions (15-min):")
        print(results.to_string(index=False))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")
        print("\nDone.")
        return results

    # ------------------------------------------------------------------
    # XGBoost / legacy path (unchanged)
    # ------------------------------------------------------------------

    # Step 3 — Select forecast rows
    print("\n-- Step 3: Selecting forecast rows")
    forecast_df = select_forecast_rows(
        live_df, horizon_hours=horizon_hours, reference_time=reference_time
    )
    print(f"Forecast rows: {len(forecast_df)}  (horizon = {horizon_hours} h)")

    if forecast_df.empty:
        print("No forecast rows available — returning empty DataFrame.")
        return pd.DataFrame(columns=["Timestamp", "Price_pred"])

    # Step 4 — Prepare features and predict
    print("\n-- Step 4: Running inference")
    features = prepare_features_for_inference(forecast_df, model=model, target_col="Price")
    predictions: np.ndarray = model.predict(features)

    # Step 5 — Resample hourly predictions to 15-minute intervals
    print("\n-- Step 5: Resampling to 15-minute intervals")
    hourly_series = pd.Series(
        predictions,
        index=pd.DatetimeIndex(forecast_df["Timestamp"].values),
        name="Price_pred",
    )
    if reference_time is not None:
        _ts = pd.Timestamp(reference_time)
        _start_15min = (_ts.tz_convert(None) if _ts.tzinfo is not None else _ts).floor("15min")
    else:
        _start_15min = pd.Timestamp.utcnow().tz_convert(None).floor("15min")
    idx_15min = pd.date_range(
        start=_start_15min,
        periods=horizon_hours * 4,
        freq="15min",
    )
    results = (
        hourly_series.reindex(hourly_series.index.union(idx_15min))
        .ffill()
        .reindex(idx_15min)
        .reset_index()
    )
    results.columns = pd.Index(["Timestamp", "Price_pred"])
    print(f"15-minute rows: {len(results)}  (expected {horizon_hours * 4})")

    print("\n-- Predictions (15-min):")
    print(results.to_string(index=False))

    # Step 6 — Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    print("\nDone.")
    return results


# Main


def main() -> None:
    """CLI entry point — parses arguments and delegates to run_inference."""
    parser = argparse.ArgumentParser(
        description="Run price inference (live or historical).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Live (auto):  python -m ors.services.price_inference.live_inference\n"
            "  LGBM dir:     python -m ors.services.price_inference.live_inference"
            " --model models/price_prediction/lgbm_recursive_single_model\n"
            "  Historical:   python -m ors.services.price_inference.live_inference"
            " --date 2025-01-15T12:00\n"
            "  CSV mode:     python -m ors.services.price_inference.live_inference"
            " --use-csv\n"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Path to model file (.joblib or .pkl) or to the LGBM model "
            "directory.  Defaults to the LGBM inference directory "
            f"({LGBM_MODEL_DIR}) with fallback to the XGBoost model "
            f"({MODEL_PATH})."
        ),
    )
    parser.add_argument(
        "--date",
        type=lambda s: datetime.fromisoformat(s).replace(tzinfo=timezone.utc),
        default=None,
        metavar="YYYY-MM-DDTHH:MM",
        help="Reference datetime in ISO format (e.g. 2025-01-15T12:00). "
        "If omitted, runs live against the current time.",
    )
    parser.add_argument(
        "--use-csv",
        action="store_true",
        default=False,
        help="Build features from the training CSVs in Data/ instead of live APIs. "
        "Produces predictions identical to those from model training.",
    )
    args = parser.parse_args()
    run_inference(model_path=args.model, reference_time=args.date, use_csv=args.use_csv)


if __name__ == "__main__":
    main()
