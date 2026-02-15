from __future__ import annotations

import argparse
from pathlib import Path

from .data_pipeline import build_merged_dataset
from .report_generator import build_report, find_project_root
from .prediction_model import train_and_save_from_dataframe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run training and generate a report.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default="xgboost")
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root or find_project_root(Path.cwd())

    merged_df = build_merged_dataset(project_root)
    run_dir = train_and_save_from_dataframe(
        project_root=project_root,
        df=merged_df,
        model_name=args.model_name,
        test_size=args.test_size,
    )

    build_report(
        project_root=project_root,
        output_path=run_dir / "model_report.pdf",
        preds_path=run_dir / "predictions.csv",
        metrics_path=run_dir / "metrics.json",
        importance_path=run_dir / "feature_importance.csv",
        model_metadata_path=run_dir / "metadata.json",
        test_size=args.test_size,
        model_name=args.model_name,
        merged_df=merged_df,
    )

    print("Pipeline completed. Report saved to:", run_dir / "model_report.pdf")


if __name__ == "__main__":
    main()
