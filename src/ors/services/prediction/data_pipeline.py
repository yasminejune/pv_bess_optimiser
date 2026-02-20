"""Data pipeline utilities for loading and merging training datasets."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ors.etl import etl


def load_source_data(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and standardise raw CSV source data from the Data directory.

    Args:
        project_root: Absolute path to the project root directory containing the Data/ folder.

    Returns:
        Tuple of (price_data, weather_data, sun_data) DataFrames with a normalised Timestamp column.

    Raises:
        FileNotFoundError: If any of the expected CSV files are absent from Data/.

    """
    data_dir = project_root / "Data"
    historical_daily_path = data_dir / "historical_daily_2025.csv"
    historical_hourly_path = data_dir / "historical_hourly_2025.csv"
    price_data_path = data_dir / "price_data.csv"

    if not historical_daily_path.exists():
        raise FileNotFoundError("historical_daily_2025.csv not found in Data/.")
    if not historical_hourly_path.exists():
        raise FileNotFoundError("historical_hourly_2025.csv not found in Data/.")
    if not price_data_path.exists():
        raise FileNotFoundError("price_data.csv not found in Data/.")

    price_data = pd.read_csv(price_data_path)
    weather_data = pd.read_csv(historical_hourly_path)
    sun_data = pd.read_csv(historical_daily_path)

    price_data = etl.standardize_timestamp_column(price_data, ["timestamp", "ts_utc", "Timestamp"])
    weather_data = etl.standardize_timestamp_column(weather_data, ["timestamp_utc", "Timestamp"])
    sun_data = etl.standardize_timestamp_column(sun_data, ["date_utc", "Timestamp"])

    return price_data, weather_data, sun_data


def build_merged_dataset(project_root: Path) -> pd.DataFrame:
    """Load source data, run the ETL pipeline, and return a merged time-sorted DataFrame.

    Args:
        project_root: Absolute path to the project root directory containing the Data/ folder.

    Returns:
        Merged and sorted DataFrame ready for feature engineering and model training.

    """
    price_data, weather_data, sun_data = load_source_data(project_root)
    merged = etl.preprocess_merge(price_data, weather_data, sun_data)
    return merged.sort_values("Timestamp").reset_index(drop=True)
