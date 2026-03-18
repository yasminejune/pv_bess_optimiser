"""Data pipeline utilities for loading and merging training datasets.

Implements the same preprocessing as the LightGBM notebook
(xgboost_price_forecast_direct.ipynb):
  - 15-min price reindexing
  - Hourly weather interpolation to 15-min
  - Daily weather forward-fill + Solar_intensity feature
  - Price outlier patching (±2 std, time-slot median replacement)
  - Time / cyclic / calendar feature engineering
"""

from __future__ import annotations

from pathlib import Path

import holidays
import numpy as np
import pandas as pd

_HOURLY_KEEP = [
    "temperature_2m",
    "relative_humidity_2m",
    "dew_point_2m",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "shortwave_radiation",
    "direct_radiation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "surface_pressure",
    "weather_code",
]

_EXCLUDED_DAILY = {"Timestamp", "sunrise", "sunset", "daylight_duration", "sunshine_duration"}


def load_source_data(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load raw CSV source data from the Data directory.

    Args:
        project_root: Absolute path to the project root directory.

    Returns:
        Tuple of (price_df, hourly_weather_df, daily_weather_df).

    Raises:
        FileNotFoundError: If any expected CSV files are absent.

    """
    data_dir = project_root / "Data" if (project_root / "Data").exists() else project_root / "data"

    price_path = data_dir / "price_data_rotated_2d.csv"
    hourly_path = data_dir / "historical_hourly_2023_2025.csv"
    daily_path = data_dir / "historical_daily_2023_2025.csv"

    if not price_path.exists():
        raise FileNotFoundError(f"Price data not found: {price_path}")
    if not hourly_path.exists():
        raise FileNotFoundError(f"Hourly weather data not found: {hourly_path}")
    if not daily_path.exists():
        raise FileNotFoundError(f"Daily weather data not found: {daily_path}")

    return pd.read_csv(price_path), pd.read_csv(hourly_path), pd.read_csv(daily_path)


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return the ordered feature column list from a preprocessed DataFrame.

    Excludes all non-feature bookkeeping columns (Timestamp, price, price_raw,
    is_price_patched, time_idx).

    Args:
        df: DataFrame produced by :func:`build_merged_dataset`.

    Returns:
        Ordered list of feature column names present in *df*.

    """
    non_feature = {"Timestamp", "price", "price_raw", "is_price_patched", "time_idx"}
    return [c for c in df.columns if c not in non_feature]


def preprocess_raw_data(
    price_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    daily_df: pd.DataFrame,
    drop_missing_price: bool = True,
) -> pd.DataFrame:
    """Run the full notebook-aligned preprocessing pipeline on raw DataFrames.

    Accepts raw DataFrames (from CSV files or live APIs) and returns a fully
    preprocessed, feature-engineered DataFrame.  Extracted from
    :func:`build_merged_dataset` so that the live inference pipeline can reuse
    the identical transformation sequence.

    Steps mirror the LightGBM notebook exactly:
    1. Reindex price data to a continuous 15-min frequency.
    2. Interpolate hourly weather features to 15-min.
    3. Forward-fill daily weather features to 15-min; compute Solar_intensity.
    4. Fill remaining NaNs; optionally drop rows without a valid price.
    5. Patch price outliers (±2 std) using time-of-day slot medians (rows with
       a valid price only).
    6. Engineer time, cyclic, and UK-holiday calendar features.

    Args:
        price_df: Raw price DataFrame.  Must contain a timestamp column
            (``timestamp`` or ``Timestamp``) and a ``price`` column.
        weather_df: Raw hourly weather DataFrame (one row per hour).  Must
            contain a timestamp column (``Timestamp`` or ``timestamp_utc``).
        daily_df: Raw daily weather DataFrame.  Must contain a date column
            (``date_utc`` or ``Timestamp``).
        drop_missing_price: If ``True`` (default), rows without a valid price
            are dropped after merging.  Pass ``False`` when the DataFrame
            includes future rows (no real price available yet) that must be
            kept for the exogenous features used during inference.

    Returns:
        Fully preprocessed and feature-engineered DataFrame sorted by Timestamp.

    """
    # ------------------------------------------------------------------ #
    # 1. Price data – select base columns, reindex to 15-min              #
    # ------------------------------------------------------------------ #
    ts_col = "timestamp" if "timestamp" in price_df.columns else "Timestamp"
    price_df = price_df.copy()
    price_df[ts_col] = pd.to_datetime(price_df[ts_col], utc=True, errors="coerce").dt.tz_convert(
        None
    )
    price_df = price_df.rename(columns={ts_col: "Timestamp"})
    price_df = price_df.sort_values("Timestamp").drop_duplicates("Timestamp")

    base_cols = [
        c
        for c in [
            "Timestamp",
            "price",
            "demand_itsdo",
            "demand_indo",
            "demand_inddem",
            "demand_forecast",
            "wind_generation",
            "solar_generation",
            "margin_daily_forecast",
        ]
        if c in price_df.columns
    ]
    price_df = price_df[base_cols]

    full_index = pd.date_range(
        price_df["Timestamp"].min(), price_df["Timestamp"].max(), freq="15min"
    )
    price_df = (
        price_df.set_index("Timestamp").reindex(full_index).rename_axis("Timestamp").reset_index()
    )

    # ------------------------------------------------------------------ #
    # 2. Hourly weather → 15-min interpolation                           #
    # ------------------------------------------------------------------ #
    weather_df = weather_df.copy()
    w_ts = "Timestamp" if "Timestamp" in weather_df.columns else "timestamp_utc"
    weather_df[w_ts] = pd.to_datetime(weather_df[w_ts], utc=True, errors="coerce").dt.tz_convert(
        None
    )
    weather_df = weather_df.rename(columns={w_ts: "WeatherTimestamp"}).sort_values(
        "WeatherTimestamp"
    )

    hourly_keep = [c for c in _HOURLY_KEEP if c in weather_df.columns]
    weather_df = weather_df[["WeatherTimestamp"] + hourly_keep].set_index("WeatherTimestamp")

    weather_15min = weather_df.reindex(full_index)

    numeric_weather = [c for c in weather_15min.columns if c != "weather_code"]
    if numeric_weather:
        weather_15min[numeric_weather] = weather_15min[numeric_weather].interpolate(
            method="time", limit_direction="both"
        )
    if "weather_code" in weather_15min.columns:
        weather_15min["weather_code"] = weather_15min["weather_code"].ffill().bfill()

    weather_15min = weather_15min.reset_index().rename(columns={"index": "Timestamp"})
    df = price_df.merge(weather_15min, on="Timestamp", how="left")

    # ------------------------------------------------------------------ #
    # 3. Daily weather → 15-min forward-fill; Solar_intensity            #
    # ------------------------------------------------------------------ #
    daily_df = daily_df.copy()
    d_ts = "date_utc" if "date_utc" in daily_df.columns else "Timestamp"
    daily_df[d_ts] = pd.to_datetime(daily_df[d_ts], utc=True, errors="coerce").dt.tz_convert(None)
    daily_df = daily_df.rename(columns={d_ts: "Timestamp"}).sort_values("Timestamp")

    if {"sunrise", "daylight_duration"}.issubset(daily_df.columns):
        sunrise = pd.to_datetime(daily_df["sunrise"], errors="coerce").dt.tz_localize(None)
        day_length_hours = daily_df["daylight_duration"] / 3600.0
        safe_day_length = day_length_hours.replace(0, np.nan)
        solar_noon_dt = sunrise + pd.to_timedelta(day_length_hours / 2, unit="h")
        hours_from_noon = (daily_df["Timestamp"] - solar_noon_dt) / np.timedelta64(1, "h")
        daily_df["Solar_intensity"] = np.maximum(
            0, np.cos(hours_from_noon * np.pi / safe_day_length)
        ).fillna(0.0)

    daily_numeric = [
        c
        for c in daily_df.columns
        if c not in _EXCLUDED_DAILY and pd.api.types.is_numeric_dtype(daily_df[c])
    ]
    daily_features = (
        daily_df[["Timestamp"] + daily_numeric].drop_duplicates("Timestamp").set_index("Timestamp")
    )
    daily_15min = (
        daily_features.reindex(full_index)
        .ffill()
        .bfill()
        .reset_index()
        .rename(columns={"index": "Timestamp"})
    )
    df = df.merge(daily_15min, on="Timestamp", how="left")

    # ------------------------------------------------------------------ #
    # 4. Fill remaining NaNs; optionally drop rows without a valid price  #
    # ------------------------------------------------------------------ #
    for col in df.columns:
        if col == "Timestamp":
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            # Only interpolate non-price columns unconditionally; price NaNs in
            # future rows must remain so callers can distinguish past/future.
            if col == "price":
                continue
            df[col] = df[col].interpolate(limit_direction="both").ffill().bfill()

    df = df.sort_values("Timestamp").reset_index(drop=True)

    if drop_missing_price:
        df = df.dropna(subset=["Timestamp", "price"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # 5. Outlier price patching (±2 std → time-slot median replacement)  #
    # ------------------------------------------------------------------ #
    # Only operate on rows that have a real price value.
    has_price = df["price"].notna()

    price_mean = df.loc[has_price, "price"].mean()
    price_std = df.loc[has_price, "price"].std()
    df["price_raw"] = df["price"]
    df["is_price_patched"] = 0

    if price_std > 0:
        cap_high = price_mean + 2 * price_std
        cap_low = price_mean - 2 * price_std
        bad_mask = has_price & ((df["price"] > cap_high) | (df["price"] < cap_low))
        good_mask = has_price & ~bad_mask

        slot_median = (
            df.loc[good_mask].groupby(df.loc[good_mask, "Timestamp"].dt.time)["price"].median()
        )
        replacement = df.loc[bad_mask, "Timestamp"].dt.time.map(slot_median)
        fallback = (
            float(df.loc[good_mask, "price"].median())
            if good_mask.any()
            else float(df.loc[has_price, "price"].median())
        )
        replacement = replacement.fillna(fallback)
        df.loc[bad_mask, "price"] = replacement.values
        df.loc[bad_mask, "is_price_patched"] = 1

    # ------------------------------------------------------------------ #
    # 6. Feature engineering                                              #
    # ------------------------------------------------------------------ #
    uk_holidays = holidays.CountryHoliday("UK")

    df["hour"] = df["Timestamp"].dt.hour.astype(int)
    df["quarter_hour"] = (df["Timestamp"].dt.minute // 15).astype(int)
    df["quarter_of_day"] = (
        (df["Timestamp"].dt.hour * 60 + df["Timestamp"].dt.minute) // 15
    ).astype(int)
    df["day_of_week"] = df["Timestamp"].dt.dayofweek.astype(int)
    df["is_weekend"] = (df["Timestamp"].dt.dayofweek >= 5).astype(int)
    df["is_holiday"] = df["Timestamp"].dt.date.isin(uk_holidays).astype(int)
    df["is_working_day"] = ((df["is_weekend"] == 0) & (df["is_holiday"] == 0)).astype(int)

    minute_of_day = df["Timestamp"].dt.hour * 60 + df["Timestamp"].dt.minute
    df["tod_sin"] = np.sin(2 * np.pi * minute_of_day / 1440.0)
    df["tod_cos"] = np.cos(2 * np.pi * minute_of_day / 1440.0)
    df["dow_sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.dayofweek / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.dayofweek / 7.0)

    df["time_idx"] = np.arange(len(df), dtype=np.int64)

    # Cast all feature columns to float
    for col in get_feature_cols(df):
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(float)

    return df


def build_merged_dataset(project_root: Path) -> pd.DataFrame:
    """Run the full notebook-aligned preprocessing pipeline.

    Steps mirror the LightGBM notebook exactly:
    1. Load and reindex price data to 15-min frequency.
    2. Interpolate hourly weather features to 15-min.
    3. Forward-fill daily weather features to 15-min; compute Solar_intensity.
    4. Fill remaining NaNs; drop rows without a valid price.
    5. Patch price outliers (±2 std) using time-of-day slot medians.
    6. Engineer time, cyclic, and UK-holiday calendar features.

    Args:
        project_root: Absolute path to the project root directory.

    Returns:
        Fully preprocessed and feature-engineered DataFrame sorted by Timestamp.

    """
    price_df, weather_df, daily_df = load_source_data(project_root)
    return preprocess_raw_data(price_df, weather_df, daily_df, drop_missing_price=True)
