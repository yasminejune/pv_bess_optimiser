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

from .data_pipeline import build_merged_dataset

if TYPE_CHECKING:
    from xgboost import XGBRegressor


def resolve_target_column(df: pd.DataFrame, target_col: str) -> str:
    """Return the name of the target column present in *df*, trying fallbacks if needed.

    Args:
        df: DataFrame to search for the target column.
        target_col: Preferred target column name.

    Returns:
        The first matching column name found in *df*.

    Raises:
        ValueError: If neither *target_col* nor any fallback candidate is in *df*.

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
        df: Merged DataFrame containing features and a target column.
        target_col: Name of the column to use as the prediction target.

    Returns:
        Tuple of ``(features, target)`` where *features* is a numeric DataFrame
        and *target* is a float Series.

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
        features: Feature DataFrame ordered by time.
        target: Corresponding target Series.
        test_size: Proportion of data to assign to the test set (between 0 and 1).

    Returns:
        Tuple of ``(x_train, x_test, y_train, y_test)``.

    Raises:
        ValueError: If *test_size* is not strictly between 0 and 1.

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
        x_train: Training feature DataFrame.
        y_train: Training target Series.
        random_state: Random seed for reproducibility.
        params: Optional dictionary of hyperparameters to override the defaults.

    Returns:
        Fitted :class:`xgboost.XGBRegressor` instance.

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


def train_and_save_from_dataframe(
    project_root: Path,
    df: pd.DataFrame,
    model_name: str = "xgboost",
    test_size: float = 0.2,
) -> Path:
    """Train an XGBoost model on *df* and persist all artifacts to a run directory.

    Args:
        project_root: Absolute path to the project root directory.
        df: Merged DataFrame containing features, a ``Timestamp`` column, and a
            ``Price`` target column.
        model_name: Base name used when creating the run directory.
        test_size: Fraction of data reserved for the test set.

    Returns:
        Path to the run directory containing the saved model and artifacts.

    """
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
