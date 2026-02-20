"""PDF report generation for XGBoost model evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from .data_pipeline import build_merged_dataset
from .prediction_model import prepare_features, resolve_target_column, time_based_split


def find_project_root(start: Path) -> Path:
    """Locate the project root by searching upward for a directory with ``Data/`` and ``src/``.

    Args:
        start: Directory path from which to begin the upward search.

    Returns:
        Path to the first ancestor directory that contains both ``Data/`` and ``src/``.

    Raises:
        FileNotFoundError: If no suitable project root is found.

    """
    for parent in [start] + list(start.parents):
        if (parent / "Data").exists() and (parent / "src").exists():
            return parent
    raise FileNotFoundError("Project root with Data/ and src/ not found.")


def find_latest_run_dir(models_dir: Path) -> Path | None:
    """Return the most recently modified subdirectory inside *models_dir*.

    Args:
        models_dir: Directory containing individual model run subdirectories.

    Returns:
        Path to the most recently modified run directory, or ``None`` if the
        directory does not exist or contains no subdirectories.

    """
    if not models_dir.exists():
        return None
    candidates = [p for p in models_dir.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_metrics(metrics_path: Path) -> dict[str, float]:
    """Load evaluation metrics from a JSON file.

    Args:
        metrics_path: Path to the metrics JSON file.

    Returns:
        Dictionary mapping metric names to float values, or an empty dict if the
        file does not exist.

    """
    if not metrics_path.exists():
        return {}
    data = json.loads(metrics_path.read_text())
    return {str(k): float(v) for k, v in data.items()}


def load_feature_importance(path: Path) -> pd.DataFrame:
    """Load a feature importance CSV into a DataFrame.

    Args:
        path: Path to the feature importance CSV file.

    Returns:
        DataFrame with ``feature`` and ``importance`` columns, or an empty
        DataFrame with those columns if the file does not exist.

    """
    if not path.exists():
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.read_csv(path)


def load_model_metadata(path: Path | None) -> dict[str, Any]:
    """Load model metadata from a JSON file.

    Args:
        path: Path to the metadata JSON file, or ``None``.

    Returns:
        Dictionary of metadata values, or an empty dict if *path* is ``None`` or
        the file does not exist.

    """
    if path is None or not path.exists():
        return {}
    return cast(dict[str, Any], json.loads(path.read_text()))


def create_text_page(pdf: PdfPages, title: str, lines: list[str]) -> None:
    """Append a formatted text page to a PDF.

    Args:
        pdf: Open :class:`~matplotlib.backends.backend_pdf.PdfPages` object.
        title: Bold title rendered at the top of the page.
        lines: List of text strings to render line by line below the title.

    """
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    fig.text(0.08, 0.95, title, fontsize=16, weight="bold")
    y = 0.9
    for line in lines:
        fig.text(0.08, y, line, fontsize=11)
        y -= 0.03
        if y < 0.05:
            pdf.savefig(fig)
            plt.close(fig)
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")
            y = 0.95
    pdf.savefig(fig)
    plt.close(fig)


def create_table_page(pdf: PdfPages, title: str, df: pd.DataFrame) -> None:
    """Append a DataFrame table page to a PDF.

    Args:
        pdf: Open :class:`~matplotlib.backends.backend_pdf.PdfPages` object.
        title: Bold title rendered at the top of the page.
        df: DataFrame whose values and column headers are rendered as a table.

    """
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    fig.text(0.08, 0.95, title, fontsize=16, weight="bold")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.2)
    pdf.savefig(fig)
    plt.close(fig)


def plot_actual_vs_predicted(df: pd.DataFrame) -> plt.Figure:
    """Create a line chart comparing actual and predicted prices over the test window.

    Args:
        df: DataFrame with ``Timestamp``, ``Price_true``, and ``Price_pred`` columns.

    Returns:
        Matplotlib Figure object.

    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(df["Timestamp"], df["Price_true"], label="Actual", linewidth=1.5)
    ax.plot(df["Timestamp"], df["Price_pred"], label="Predicted", linewidth=1.2, alpha=0.85)
    ax.set_title("Actual vs Predicted Price (Test Window)")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_residuals(df: pd.DataFrame) -> plt.Figure:
    """Create a time-series plot of prediction residuals.

    Args:
        df: DataFrame with ``Timestamp`` and ``Residual`` columns.

    Returns:
        Matplotlib Figure object.

    """
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df["Timestamp"], df["Residual"], linewidth=1.0)
    ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
    ax.set_title("Residuals Over Time")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Prediction Error")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_residual_distribution(df: pd.DataFrame) -> plt.Figure:
    """Create a histogram of the residual distribution.

    Args:
        df: DataFrame with a ``Residual`` column.

    Returns:
        Matplotlib Figure object.

    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(df["Residual"].dropna(), bins=40, edgecolor="white", alpha=0.85)
    ax.set_title("Residual Distribution")
    ax.set_xlabel("Prediction Error")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 20) -> plt.Figure:
    """Create a horizontal bar chart of the top-*n* most important features.

    Args:
        importance_df: DataFrame with ``feature`` and ``importance`` columns sorted
            in descending order of importance.
        top_n: Maximum number of features to display.

    Returns:
        Matplotlib Figure object.

    """
    plot_df = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#4C78A8")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(True, axis="x", alpha=0.2)
    fig.tight_layout()
    return fig


def build_report(
    project_root: Path,
    output_path: Path,
    preds_path: Path,
    metrics_path: Path,
    importance_path: Path,
    model_metadata_path: Path | None,
    test_size: float,
    model_name: str,
    target_col: str = "Price",
    merged_df: pd.DataFrame | None = None,
) -> None:
    """Build and save a multi-page PDF model evaluation report.

    Generates pages covering summary statistics, evaluation metrics, feature list,
    preprocessing notes, and visualisations (actual vs predicted, residuals, feature
    importances).

    Args:
        project_root: Absolute path to the project root directory.
        output_path: Destination file path for the PDF report.
        preds_path: Path to the CSV file containing model predictions.
        metrics_path: Path to the JSON file containing evaluation metrics.
        importance_path: Path to the CSV file containing feature importances.
        model_metadata_path: Path to the JSON metadata file, or ``None``.
        test_size: Fraction of data that was used as the test set.
        model_name: Human-readable model name shown in the report.
        target_col: Name of the target column in the merged dataset.
        merged_df: Pre-loaded merged DataFrame; loaded from disk when ``None``.

    """
    preds = pd.read_csv(preds_path, parse_dates=["Timestamp"])
    metrics = load_metrics(metrics_path)
    importance_df = load_feature_importance(importance_path)
    metadata = load_model_metadata(model_metadata_path)

    if merged_df is None:
        merged_df = build_merged_dataset(project_root)

    resolved_target = resolve_target_column(merged_df, target_col)
    features, target = prepare_features(merged_df, target_col=resolved_target)
    x_train, x_test, y_train, y_test = time_based_split(features, target, test_size=test_size)
    split_idx = len(x_train)

    test_df = merged_df.iloc[split_idx:][["Timestamp", resolved_target]].copy()
    test_df = test_df.rename(columns={resolved_target: "Price_true"}).reset_index(drop=True)

    if "Price_true" in preds.columns:
        analysis_df = preds.copy()
    else:
        analysis_df = preds.merge(test_df, on="Timestamp", how="left")
    analysis_df = analysis_df.sort_values("Timestamp").reset_index(drop=True)
    analysis_df["Residual"] = analysis_df["Price_pred"] - analysis_df["Price_true"]

    encoding_notes = [
        "Timestamp dropped from model features.",
        "Boolean features cast to integer (0/1).",
        "Missing values filled with median per feature.",
    ]

    report_lines = [
        f"Model: {model_name}",
        f"Rows total: {len(merged_df)}",
        f"Train rows: {len(x_train)}",
        f"Test rows: {len(x_test)}",
        f"Test size: {test_size:.2f}",
        f"Time range: {merged_df['Timestamp'].min()} to {merged_df['Timestamp'].max()}",
    ]

    if metrics:
        report_lines.append(" ")
        report_lines.append("Key metrics:")
        for key in ["mae", "rmse", "mape", "r2"]:
            if key in metrics:
                report_lines.append(f"- {key}: {metrics[key]:.4f}")

    if metadata:
        report_lines.append(" ")
        report_lines.append("Model metadata:")
        for key, value in metadata.items():
            report_lines.append(f"- {key}: {value}")

    metrics_df = (
        pd.DataFrame([metrics]) if metrics else pd.DataFrame(columns=["mae", "rmse", "mape", "r2"])
    )
    features_df = pd.DataFrame(
        {
            "feature": list(features.columns),
            "type": [str(dtype) for dtype in features.dtypes],
        }
    )

    with PdfPages(output_path) as pdf:
        create_text_page(pdf, "Model Report Summary", report_lines)
        create_table_page(pdf, "Evaluation Metrics", metrics_df)
        create_table_page(pdf, "Feature List", features_df)
        create_text_page(pdf, "Encoding and Preprocessing", encoding_notes)

        fig_actual_pred = plot_actual_vs_predicted(analysis_df)
        pdf.savefig(fig_actual_pred)
        plt.close(fig_actual_pred)

        fig_residuals = plot_residuals(analysis_df)
        pdf.savefig(fig_residuals)
        plt.close(fig_residuals)

        fig_hist = plot_residual_distribution(analysis_df)
        pdf.savefig(fig_hist)
        plt.close(fig_hist)

        if not importance_df.empty:
            fig_importance = plot_feature_importance(importance_df)
            pdf.savefig(fig_importance)
            plt.close(fig_importance)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the report generator.

    Returns:
        Parsed argument namespace.

    """
    parser = argparse.ArgumentParser(description="Generate a PDF model report.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preds", type=Path, default=None)
    parser.add_argument("--metrics", type=Path, default=None)
    parser.add_argument("--importance", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--model-name", type=str, default="XGBoost")
    return parser.parse_args()


def main() -> None:
    """Generate a PDF model evaluation report from the command line.

    Raises:
        FileNotFoundError: If a specified run directory does not exist.

    """
    args = parse_args()
    project_root = args.project_root or find_project_root(Path.cwd())
    prediction_dir = project_root / "Prediction"
    models_dir = prediction_dir / "Models"

    run_dir = args.run_dir
    if run_dir is None:
        run_dir = find_latest_run_dir(models_dir)

    if run_dir is not None and not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    if run_dir is not None:
        output_path = args.output or (run_dir / "model_report.pdf")
        preds_path = args.preds or (run_dir / "predictions.csv")
        metrics_path = args.metrics or (run_dir / "metrics.json")
        importance_path = args.importance or (run_dir / "feature_importance.csv")
        metadata_path = args.metadata or (run_dir / "metadata.json")
    else:
        output_path = args.output or (prediction_dir / "model_report.pdf")
        preds_path = args.preds or (prediction_dir / "xgb_predictions.csv")
        metrics_path = args.metrics or (prediction_dir / "xgb_metrics.json")
        importance_path = args.importance or (prediction_dir / "xgb_feature_importance.csv")
        metadata_path = args.metadata

    build_report(
        project_root=project_root,
        output_path=output_path,
        preds_path=preds_path,
        metrics_path=metrics_path,
        importance_path=importance_path,
        model_metadata_path=metadata_path,
        test_size=args.test_size,
        model_name=args.model_name,
    )

    print("Report saved to:", output_path)


if __name__ == "__main__":
    main()
