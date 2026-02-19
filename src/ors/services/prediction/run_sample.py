from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Support both direct execution and package import
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ors.services.prediction.prediction_model import train_and_save_from_dataframe
else:
    from .prediction_model import train_and_save_from_dataframe


def main() -> None:
    project_root = Path(__file__).resolve().parents[4]
    sample_path = project_root / "Data" / "merged_dataset.csv"

    df = pd.read_csv(sample_path, parse_dates=["Timestamp"])
    run_dir = train_and_save_from_dataframe(
        project_root=project_root,
        df=df,
        model_name="xgboost_sample",
        test_size=0.25,
    )

    print("Sample run saved to:", run_dir)


if __name__ == "__main__":
    main()
