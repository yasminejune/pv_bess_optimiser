"""XGBoost model training, evaluation, and persistence utilities."""

from __future__ import annotations

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data_pipeline import build_merged_dataset, get_feature_cols

if TYPE_CHECKING:
    from skforecast.recursive import ForecasterRecursive
    from xgboost import XGBRegressor


def resolve_target_column(df: pd.DataFrame, target_col: str) -> str:
    """Return the name of the target column present in *df*, trying fallbacks if needed.

    Args:
        df (pd.DataFrame): DataFrame to search for the target column
        target_col (str): Preferred target column name

    Returns:
        str: The first matching column name found in *df*

    Raises:
        ValueError: If neither *target_col* nor any fallback candidate is in *df*

    """
    if target_col in df.columns:
        return target_col
    for candidate in ["price", "target_price", "Price"]:
        if candidate in df.columns:
            return candidate
    raise ValueError(f"Target column '{target_col}' not found in merged dataset.")


def prepare_features(df: pd.DataFrame, target_col: str = "Price") -> tuple[pd.DataFrame, pd.Series]:
    """Extract numeric feature matrix and target vector from a merged DataFrame.

    Drops the target column, ``Timestamp``, and non-numeric columns; casts booleans
    to integers; fills remaining ``NaN`` values with column medians.

    Args:
        df (pd.DataFrame): Merged DataFrame containing features and a target column
        target_col (str): Name of the column to use as the prediction target

    Returns:
        tuple[pd.DataFrame, pd.Series]: Tuple of ``(features, target)`` where *features* is a numeric DataFrame
            and *target* is a float Series

    """
    resolved_target = resolve_target_column(df, target_col)

    features = df.drop(columns=[resolved_target])
    if "Timestamp" in features.columns:
        features = features.drop(columns=["Timestamp"])

    numeric_cols = features.select_dtypes(include=["number", "bool"]).columns
    features = features[numeric_cols].copy()
    for col in features.columns:
        if features[col].dtype == "bool":
            features[col] = features[col].astype(int)

    for col in features.columns:
        if features[col].isna().any():
            features[col] = features[col].fillna(features[col].median())

    target = df[resolved_target].astype(float)
    return features, target


def prepare_features_for_inference(df: pd.DataFrame, target_col: str = "Price") -> pd.DataFrame:
    """Extract numeric feature matrix from a DataFrame, tolerating a missing target column.

    Drops the target column if present, along with ``Timestamp`` and non-numeric columns;
    casts booleans to integers; fills ``NaN`` values with column medians.

    Args:
        df: DataFrame that may or may not include a target column.
        target_col: Name of the target column to drop if present.

    Returns:
        Numeric feature DataFrame suitable for model inference.

    """
    resolved_target = None
    for candidate in [target_col, "price", "target_price", "Price"]:
        if candidate in df.columns:
            resolved_target = candidate
            break
    features = df.copy()
    if resolved_target is not None:
        features = features.drop(columns=[resolved_target])
    if "Timestamp" in features.columns:
        features = features.drop(columns=["Timestamp"])

    numeric_cols = features.select_dtypes(include=["number", "bool"]).columns
    features = features[numeric_cols].copy()
    for col in features.columns:
        if features[col].dtype == "bool":
            features[col] = features[col].astype(int)

    for col in features.columns:
        if features[col].isna().any():
            features[col] = features[col].fillna(features[col].median())

    return features


def time_based_split(
    features: pd.DataFrame, target: pd.Series, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split features and target into chronological train/test sets.

    Args:
        features (pd.DataFrame): Feature DataFrame ordered by time
        target (pd.Series): Corresponding target Series
        test_size (float): Proportion of data to assign to the test set (between 0 and 1)

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: Tuple of ``(x_train, x_test, y_train, y_test)``

    Raises:
        ValueError: If *test_size* is not strictly between 0 and 1

    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    split_idx = int(len(features) * (1 - test_size))
    x_train = features.iloc[:split_idx]
    x_test = features.iloc[split_idx:]
    y_train = target.iloc[:split_idx]
    y_test = target.iloc[split_idx:]
    return x_train, x_test, y_train, y_test


def train_xgb_regressor(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    params: dict[str, Any] | None = None,
) -> XGBRegressor:
    """Train an XGBoost regressor with sensible defaults and optional overrides.

    Args:
        x_train (pd.DataFrame): Training feature DataFrame
        y_train (pd.Series): Training target Series
        random_state (int): Random seed for reproducibility
        params (dict[str, Any] | None): Optional dictionary of hyperparameters to override the defaults

    Returns:
        XGBRegressor: Fitted :class:`xgboost.XGBRegressor` instance

    """
    from xgboost import XGBRegressor

    base_params = {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "objective": "reg:squarederror",
        "random_state": random_state,
        "n_jobs": 4,
    }
    if params:
        base_params.update(params)
        base_params["random_state"] = random_state

    model = XGBRegressor(**base_params)
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: XGBRegressor, x_test: pd.DataFrame, y_test: pd.Series
) -> dict[str, float]:
    """Compute regression evaluation metrics for a trained model on held-out data.

    Args:
        model: Trained XGBRegressor instance.
        x_test: Test feature DataFrame.
        y_test: True target values for the test set.

    Returns:
        Dictionary with keys ``mae``, ``rmse``, ``mape``, and ``r2``.

    """
    preds = model.predict(x_test)
    safe_denom = np.where(np.abs(y_test) < 1e-8, np.nan, y_test)
    raw_mape = float(np.nanmean(np.abs((y_test - preds) / safe_denom)) * 100)
    mape = raw_mape if not np.isnan(raw_mape) else float("inf")
    return {
        "mae": float(mean_absolute_error(y_test, preds)),
        "rmse": float(mean_squared_error(y_test, preds)),
        "mape": mape,
        "r2": float(r2_score(y_test, preds)),
    }


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    """Serialise an evaluation metrics dictionary to a JSON file.

    Args:
        metrics: Dictionary of metric names to float values.
        output_path: Destination file path for the JSON output.

    """
    output_path.write_text(json.dumps(metrics, indent=2))


def save_predictions(
    timestamps: pd.Series, y_true: pd.Series, y_pred: np.ndarray, output_path: Path
) -> None:
    """Save ground-truth and predicted values alongside their timestamps to CSV.

    Args:
        timestamps: Timestamp Series aligned to *y_true* and *y_pred*.
        y_true: True target values.
        y_pred: Predicted target values.
        output_path: Destination file path for the CSV output.

    """
    output = pd.DataFrame(
        {
            "Timestamp": timestamps.values,
            "Price_true": y_true.values,
            "Price_pred": y_pred,
        }
    )
    output.to_csv(output_path, index=False)


def save_feature_importance(model: XGBRegressor, features: pd.DataFrame, output_path: Path) -> None:
    """Save a sorted feature importance table to CSV.

    Args:
        model: Trained XGBRegressor exposing a ``feature_importances_`` attribute.
        features: Feature DataFrame used during training (provides column names).
        output_path: Destination file path for the CSV output.

    """
    importance = model.feature_importances_
    table = (
        pd.DataFrame({"feature": features.columns, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    table.to_csv(output_path, index=False)


def create_model_run_dir(project_root: Path, model_name: str) -> Path:
    """Create a unique, timestamped directory under ``Prediction/Models/`` for a run.

    Args:
        project_root: Absolute path to the project root directory.
        model_name: Base name used as a prefix for the run directory.

    Returns:
        Path to the newly created run directory.

    """
    models_dir = project_root / "Prediction" / "Models"
    models_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{model_name}_{timestamp}"
    run_dir = models_dir / base_name
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        return run_dir

    index = 1
    while True:
        candidate = models_dir / f"{base_name}_{index}"
        if not candidate.exists():
            candidate.mkdir(parents=True)
            return candidate
        index += 1


def predict_prices(model: XGBRegressor, df: pd.DataFrame, target_col: str = "Price") -> np.ndarray:
    """Generate price predictions for a DataFrame using a trained model.

    Args:
        model: Trained XGBRegressor with a ``predict`` method.
        df: DataFrame containing the same features used during training.
        target_col: Name of the target column to exclude from features if present.

    Returns:
        NumPy array of predicted price values.

    """
    features = prepare_features_for_inference(df, target_col=target_col)
    return cast(np.ndarray, model.predict(features))


def _build_lgbm_recursive(n_estimators: int = 850) -> ForecasterRecursive:
    """Build a LightGBM ForecasterRecursive with notebook-aligned hyperparameters.

    Args:
        n_estimators: Number of boosting rounds.

    Returns:
        Configured but unfitted :class:`skforecast.recursive.ForecasterRecursive`.

    """
    from lightgbm import LGBMRegressor
    from skforecast.preprocessing import RollingFeatures
    from skforecast.recursive import ForecasterRecursive

    regressor = LGBMRegressor(
        objective="mae",
        n_estimators=n_estimators,
        learning_rate=0.025,
        num_leaves=127,
        max_depth=11,
        min_child_samples=20,
        min_split_gain=0.0,
        subsample=0.95,
        colsample_bytree=0.95,
        reg_alpha=0.1,
        reg_lambda=0.8,
        max_bin=511,
        random_state=42,
        n_jobs=-1,
    )

    rolling_windows = [4, 16, 32, 96, 192, 288, 384, 576]
    rolling_stats = ["mean", "std", "min", "max"]
    rf_stats = [stat for win in rolling_windows for stat in rolling_stats]
    rf_windows = [win for win in rolling_windows for _ in rolling_stats]
    window_features = RollingFeatures(stats=rf_stats, window_sizes=rf_windows)

    recursive_lags = list(range(1, 49)) + [64, 96, 128, 192, 288, 384, 480, 576, 672]
    return ForecasterRecursive(
        regressor=regressor,
        lags=recursive_lags,
        window_features=window_features,
    )


def _train_and_save_lgbm(
    project_root: Path,
    df: pd.DataFrame,
    model_name: str = "lgbm_recursive",
) -> Path:
    """Train a LightGBM ForecasterRecursive and save all artifacts.

    Replicates the notebook training exactly:
    - 180-day rolling training window
    - Fixed val/test splits of 21 days each
    - Saves model + metadata via joblib; also writes report-compatible CSV/JSON

    Args:
        project_root: Absolute path to the project root directory.
        df: Preprocessed DataFrame from :func:`~data_pipeline.build_merged_dataset`.
        model_name: Base name used when creating the run directory.

    Returns:
        Path to the run directory containing all saved artifacts.

    """
    import joblib

    horizon = 96
    val_steps = 21 * 24 * 4
    test_steps = 21 * 24 * 4
    train_lookback_days = 180
    n_estimators = 850

    feature_cols = get_feature_cols(df)
    y_all = df["price"].astype(float).reset_index(drop=True)
    exog_all = df[feature_cols].reset_index(drop=True)

    max_idx = int(len(df) - 1)
    train_cutoff = max_idx - (val_steps + test_steps)
    val_start = train_cutoff + 1
    test_start = val_start + val_steps

    train_lookback_steps = train_lookback_days * 24 * 4
    train_start_idx = max(0, train_cutoff - train_lookback_steps + 1)

    train_y = y_all.iloc[train_start_idx : train_cutoff + 1]
    train_exog = exog_all.iloc[train_start_idx : train_cutoff + 1]

    print(f"Training LGBM recursive model (n_estimators={n_estimators}) …")
    print(f"  window: rows {train_start_idx}–{train_cutoff} ({len(train_y):,} rows)")

    model = _build_lgbm_recursive(n_estimators=n_estimators)
    model.fit(y=train_y, exog=train_exog)

    max_lag = int(getattr(model, "window_size", 672))

    def _eval_segment(start_idx: int, seg_len: int) -> tuple[np.ndarray, np.ndarray]:
        y_true_all: list[np.ndarray] = []
        y_pred_all: list[np.ndarray] = []
        end_excl = start_idx + seg_len
        for origin in range(start_idx, end_excl, horizon):
            if origin + horizon > end_excl:
                break
            if origin < max_lag:
                continue
            last_window = y_all.iloc[origin - max_lag : origin]
            exog_future = exog_all.iloc[origin : origin + horizon]
            pred = model.predict(
                steps=horizon, last_window=last_window, exog=exog_future
            ).to_numpy()
            y_true_all.append(y_all.iloc[origin : origin + horizon].to_numpy())
            y_pred_all.append(pred)
        if not y_true_all:
            return np.empty(0), np.empty(0)
        return np.concatenate(y_true_all), np.concatenate(y_pred_all)

    test_true, test_pred = _eval_segment(test_start, test_steps)

    # Compute test metrics
    if len(test_true) > 0:
        mae_val = float(mean_absolute_error(test_true, test_pred))
        rmse_val = float(np.sqrt(mean_squared_error(test_true, test_pred)))
        mape_val = float(
            np.mean(np.abs((test_true - test_pred) / np.clip(np.abs(test_true), 1e-6, None))) * 100
        )
    else:
        mae_val = rmse_val = mape_val = float("nan")

    met: dict[str, float] = {"mae": mae_val, "rmse": rmse_val, "mape": mape_val}

    # Create run directory
    run_dir = create_model_run_dir(project_root, model_name)

    # Save model and metadata via joblib
    model_path = run_dir / "model.joblib"
    meta: dict = {
        "model_name": model_name,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "feature_cols": feature_cols,
        "horizon": horizon,
        "max_lag": max_lag,
        "n_estimators": n_estimators,
        "train_end_idx": train_cutoff,
        "row_count": len(df),
        "feature_count": len(feature_cols),
    }
    joblib.dump(model, model_path)
    joblib.dump(meta, run_dir / "model_meta.joblib")

    # Also write to the standard inference path used by the forecasting service
    inference_dir = project_root / "models" / "price_prediction" / "lgbm_recursive_single_model"
    inference_dir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    joblib.dump(model, inference_dir / f"recursive_single_model_{ts_str}.joblib")
    joblib.dump(meta, inference_dir / f"recursive_single_model_{ts_str}_meta.joblib")
    print("Inference model saved to:", inference_dir)

    # Save metrics JSON (report-compatible keys)
    save_metrics(met, run_dir / "metrics.json")

    # Save test predictions CSV
    n_test = len(test_true)
    test_timestamps = df.iloc[test_start : test_start + n_test]["Timestamp"]
    preds_df = pd.DataFrame(
        {
            "Timestamp": test_timestamps.values,
            "Price_true": test_true,
            "Price_pred": test_pred,
        }
    )
    preds_df.to_csv(run_dir / "predictions.csv", index=False)

    # Save feature importance
    importance_df = model.get_feature_importances()
    importance_df.to_csv(run_dir / "feature_importance.csv", index=False)

    # Save metadata JSON (scalar values only for report compatibility)
    json_meta = {k: v for k, v in meta.items() if not isinstance(v, (list, dict))}
    (run_dir / "metadata.json").write_text(json.dumps(json_meta, indent=2))

    print("Model saved to:", model_path)
    print("Metrics saved to:", run_dir / "metrics.json")
    print("Predictions saved to:", run_dir / "predictions.csv")
    print("Metrics:", met)

    return run_dir


def train_and_save_from_dataframe(
    project_root: Path,
    df: pd.DataFrame,
    model_name: str = "lgbm_recursive",
    test_size: float = 0.2,
) -> Path:
    """Train a model on *df* and persist all artifacts to a run directory.

    When *model_name* contains ``"lgbm"`` (case-insensitive) the LightGBM
    ForecasterRecursive path is used (notebook-aligned).  Otherwise the
    legacy XGBoost path is used.

    Args:
        project_root: Absolute path to the project root directory.
        df: Merged DataFrame containing features, a ``Timestamp`` column, and a
            target column (``price`` or ``Price``).
        model_name: Base name used when creating the run directory.
        test_size: Fraction of data reserved for the test set (XGBoost path only).

    Returns:
        Path to the run directory containing the saved model and artifacts.

    """
    if "lgbm" in model_name.lower():
        return _train_and_save_lgbm(project_root, df, model_name)

    features, target = prepare_features(df, target_col="Price")
    x_train, x_test, y_train, y_test = time_based_split(features, target, test_size=test_size)

    model = train_xgb_regressor(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)

    run_dir = create_model_run_dir(project_root, model_name)
    model_path = run_dir / "model.json"
    metrics_path = run_dir / "metrics.json"
    preds_path = run_dir / "predictions.csv"
    importance_path = run_dir / "feature_importance.csv"
    metadata_path = run_dir / "metadata.json"

    model.save_model(model_path)

    # Also write a .pkl copy to the standard inference path
    inference_pkl = project_root / "models" / "price_prediction" / "model.pkl"
    inference_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(inference_pkl, "wb") as fh:
        pickle.dump(model, fh)
    print("Inference model saved to:", inference_pkl)

    save_metrics(metrics, metrics_path)
    save_predictions(df.loc[y_test.index, "Timestamp"], y_test, model.predict(x_test), preds_path)
    save_feature_importance(model, features, importance_path)
    metadata = {
        "model_name": model_name,
        "run_timestamp": datetime.now().isoformat(timespec="seconds"),
        "test_size": test_size,
        "row_count": len(df),
        "feature_count": len(features.columns),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("Model saved to:", model_path)
    print("Metrics saved to:", metrics_path)
    print("Predictions saved to:", preds_path)
    print("Feature importance saved to:", importance_path)
    print("Metadata saved to:", metadata_path)
    print("Metrics:", metrics)

    return run_dir


def train_and_save(project_root: Path, model_name: str = "xgboost", test_size: float = 0.2) -> Path:
    """Build the merged dataset, train a model, and save all artifacts.

    Args:
        project_root: Absolute path to the project root directory containing ``Data/``.
        model_name: Base name used when creating the run directory.
        test_size: Fraction of data reserved for the test set.

    Returns:
        Path to the run directory containing the saved model and artifacts.

    """
    df = build_merged_dataset(project_root)
    return train_and_save_from_dataframe(
        project_root, df, model_name=model_name, test_size=test_size
    )


def main() -> None:
    """Train and save a model using the default project root layout."""
    train_and_save(Path(__file__).resolve().parents[4])


if __name__ == "__main__":
    main()
