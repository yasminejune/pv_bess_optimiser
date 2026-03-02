"""Live inference: generate next-24-hour electricity price predictions.

Based on API outputs (Weather + Price).

To switch models, pass ``--model`` on the command line or update ``MODEL_PATH``.

Run from the repo root::

    python -m ors.services.price_inference.live_inference
"""

from __future__ import annotations

import argparse
import pickle  # models will be saved in this format
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ors.services.prediction.data_pipeline import build_merged_dataset
from ors.services.price_inference.live_data_pipeline import (
    PRICE_LOOKBACK_HOURS,
    build_live_merged_dataset,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Path to the pickled model file. Update this when testing a new model.
MODEL_PATH: Path = Path("models/price_prediction/model.pkl")

# Number of hours ahead to predict.
HORIZON_HOURS: int = 24

# Lag steps — must match those used during training.
LAG_STEPS: tuple[int, ...] = (1, 2, 3, 6, 12, 24)

# Path to write the 15-minute predictions CSV consumed by the optimizer.
OUTPUT_PATH: Path = Path("data/live_price_predictions.csv")


# ---------------------------------------------------------------------------
# Feature preparation (mirrors training's prepare_features_for_inference)
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
# Model loading
# ---------------------------------------------------------------------------


def load_model(model_path: Path) -> Any:
    """Load a pickled model from disk.

    Args:
        model_path: Path to the model being used.

    Returns:
        Deserialised model object.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Update MODEL_PATH in live_inference.py to point to your .pkl file."
        )
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)
    print(f"Model loaded: {model_path}  ({type(model).__name__})")
    return model


# ---------------------------------------------------------------------------
# Forecast row selection
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


# Full pipeline (importable entry point for next modules)


def run_inference(
    model_path: Path = MODEL_PATH,
    horizon_hours: int = HORIZON_HOURS,
    lag_steps: tuple[int, ...] = LAG_STEPS,
    output_path: Path = OUTPUT_PATH,
    reference_time: datetime | None = None,
    use_csv: bool = False,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """Run the full price inference pipeline.

    Fetches data, runs ETL, loads the model, predicts, resamples to
    15-minute resolution, saves the result to CSV, and returns it.

    In live mode (reference_time=None) data is fetched up to now and the
    forecast window starts from the current hour.  In historical mode all
    fetches and the forecast window are anchored to reference_time, allowing
    replay of a past date for backtesting or evaluation.

    Args:
        model_path: Path to the pickled model file.
        horizon_hours: Number of hours ahead to predict.
        lag_steps: Lag offsets in hours — must match those used during training.
        output_path: Path to write the predictions CSV.
        reference_time: UTC datetime to treat as "now".  If None, runs live.
        use_csv: If True, build features from the training CSVs in Data/ instead
            of calling live APIs.  Produces predictions identical to those from
            model training for any timestamp covered by the training data.
        project_root: Repo root used when use_csv=True.  Defaults to four
            directories above this file.

    Returns:
        DataFrame with columns ``Timestamp`` and ``Price_pred``
        at 15-minute resolution (``horizon_hours * 4`` rows).
    """
    if use_csv:
        root = project_root or Path(__file__).resolve().parents[4]
        mode = f"CSV ({reference_time})" if reference_time is not None else "CSV (all)"
        print(f"\n-- Step 1: Building {mode} merged dataset from training CSVs")
        live_df = build_merged_dataset(root)
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

    # Step 2 — Load model
    print("\n-- Step 2: Loading model")
    model = load_model(model_path)

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
    # Start the 15-min window at the current 15-min slot, not the top of the hour.
    # e.g. at 2:20 → start at 2:15 instead of 2:00.
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
            "  Live:        python -m ors.services.price_inference.live_inference\n"
            "  Historical:  python -m ors.services.price_inference.live_inference"
            " --date 2025-01-15T12:00\n"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Path to the .pkl model file (default: %(default)s)",
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
