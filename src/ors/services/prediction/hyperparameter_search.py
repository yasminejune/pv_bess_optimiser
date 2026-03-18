"""XGBoost hyperparameter grid search with artifact persistence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import ParameterGrid

# Support both direct execution and package import
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ors.services.prediction.data_pipeline import build_merged_dataset
    from ors.services.prediction.prediction_model import (
        create_model_run_dir,
        evaluate_model,
        prepare_features,
        save_feature_importance,
        save_metrics,
        save_predictions,
        time_based_split,
        train_xgb_regressor,
    )
else:
    from .data_pipeline import build_merged_dataset
    from .prediction_model import (
        create_model_run_dir,
        evaluate_model,
        prepare_features,
        save_feature_importance,
        save_metrics,
        save_predictions,
        time_based_split,
        train_xgb_regressor,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the hyperparameter search script.

    Returns:
        Parsed argument namespace.

    """
    parser = argparse.ArgumentParser(
        description="Run an XGBoost hyperparameter search and save the best model artifacts."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="xgboost_search")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument(
        "--score-metric",
        type=str,
        choices=["rmse", "mae", "mape", "r2"],
        default="mape",
    )
    parser.add_argument("--max-evals", type=int, default=0)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def default_param_grid() -> dict[str, list[Any]]:
    """Return the default XGBoost hyperparameter search grid.

    Returns:
        Dictionary mapping hyperparameter names to lists of candidate values.

    """
    return {
        "n_estimators": [300, 500],
        "learning_rate": [0.03, 0.07],
        "max_depth": [4, 6, 8],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "min_child_weight": [1, 5],
    }


def is_better(metric_name: str, candidate: float, best: float | None) -> bool:
    """Determine whether *candidate* is an improvement over *best* for a given metric.

    For ``r2`` a higher value is better; for all other metrics a lower value is better.

    Args:
        metric_name: Name of the metric being compared (e.g. ``"rmse"`` or ``"r2"``).
        candidate: The new metric value to evaluate.
        best: The current best metric value, or ``None`` if no baseline exists yet.

    Returns:
        ``True`` if *candidate* is better than *best* (or *best* is ``None``).

    """
    if best is None:
        return True
    if metric_name == "r2":
        return candidate > best
    return candidate < best


def main() -> None:
    """Run a grid hyperparameter search and save the best model artifacts.

    Raises:
        RuntimeError: If no models were trained (empty parameter grid).

    """
    args = parse_args()
    project_root = args.project_root or (Path(__file__).resolve().parents[4])

    merged_df = build_merged_dataset(project_root)
    features, target = prepare_features(merged_df, target_col="Price")
    x_train, x_test, y_train, y_test = time_based_split(features, target, test_size=args.test_size)

    param_grid = list(ParameterGrid(default_param_grid()))
    if args.max_evals and args.max_evals > 0:
        param_grid = param_grid[: args.max_evals]

    run_dir = create_model_run_dir(project_root, args.model_name)
    results: list[dict[str, Any]] = []
    best_metric_value: float | None = None
    best_params: dict[str, Any] | None = None
    best_model: Any | None = None
    best_metrics: dict[str, float] | None = None

    for idx, params in enumerate(param_grid, start=1):
        model = train_xgb_regressor(
            x_train,
            y_train,
            random_state=args.random_state,
            params=params,
        )
        metrics = evaluate_model(model, x_test, y_test)
        score = metrics[args.score_metric]

        results.append(
            {
                "trial": idx,
                **params,
                **metrics,
                "score_metric": args.score_metric,
                "score_value": score,
            }
        )

        if is_better(args.score_metric, score, best_metric_value):
            best_metric_value = score
            best_params = params
            best_model = model
            best_metrics = metrics

    results_df = pd.DataFrame(results)
    results_path = run_dir / "search_results.csv"
    results_df.to_csv(results_path, index=False)

    if best_model is None or best_params is None or best_metrics is None:
        raise RuntimeError("No models were trained; check the parameter grid.")

    model_path = run_dir / "model.json"
    metrics_path = run_dir / "metrics.json"
    preds_path = run_dir / "predictions.csv"
    importance_path = run_dir / "feature_importance.csv"
    metadata_path = run_dir / "metadata.json"

    best_model.save_model(model_path)
    save_metrics(best_metrics, metrics_path)
    save_predictions(
        merged_df.loc[y_test.index, "Timestamp"],
        y_test,
        best_model.predict(x_test),
        preds_path,
    )
    save_feature_importance(best_model, features, importance_path)

    metadata = {
        "model_name": args.model_name,
        "score_metric": args.score_metric,
        "best_score_value": best_metric_value,
        "best_params": best_params,
        "test_size": args.test_size,
        "row_count": len(merged_df),
        "feature_count": len(features.columns),
        "search_trials": len(results),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print("Search results saved to:", results_path)
    print("Best model saved to:", model_path)
    print("Best metrics:", best_metrics)


if __name__ == "__main__":
    main()
