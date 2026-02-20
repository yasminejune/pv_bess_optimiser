"""ETL utilities for loading, transforming, and merging energy and weather datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Extract


def standardize_timestamp_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    target_column: str = "Timestamp",
) -> pd.DataFrame:
    """Rename and normalise the timestamp column of a DataFrame.

    Searches *dataframe* for the first column name found in *candidates*, renames it
    to *target_column*, coerces it to ``datetime64`` and strips any timezone info.

    Args:
        dataframe: Input DataFrame that contains a timestamp-like column.
        candidates: Ordered list of column names to search for the timestamp.
        target_column: Desired name for the resulting timestamp column.

    Returns:
        The modified DataFrame with a tz-naive ``datetime64`` timestamp column.

    Raises:
        ValueError: If none of the candidate column names are present in *dataframe*.

    """
    timestamp_col = next((col for col in candidates if col in dataframe.columns), None)
    if timestamp_col is None:
        raise ValueError(f"Missing timestamp column. Tried: {', '.join(candidates)}")
    if timestamp_col != target_column:
        dataframe.rename(columns={timestamp_col: target_column}, inplace=True)
    dataframe[target_column] = pd.to_datetime(dataframe[target_column], errors="coerce")
    dataframe[target_column] = dataframe[target_column].dt.tz_localize(None)
    return dataframe


# Time data
def generate_time_data(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    country_holidays: str,
) -> pd.DataFrame:
    """Generate an hourly time-feature DataFrame between two dates.

    Includes calendar features (hour, day, month, etc.) as well as flags for
    weekends, public holidays and working days.

    Args:
        start_date: Inclusive start of the date range.
        end_date: Inclusive end of the date range.
        country_holidays: ISO country code passed to ``holidays.CountryHoliday``.

    Returns:
        DataFrame with a ``Timestamp`` column plus derived time-feature columns.

    """
    time_data = pd.DataFrame({"Timestamp": pd.date_range(start=start_date, end=end_date, freq="h")})
    time_data["Hour"] = time_data["Timestamp"].dt.hour
    time_data["Day"] = time_data["Timestamp"].dt.day
    time_data["Month"] = time_data["Timestamp"].dt.month
    time_data["Year"] = time_data["Timestamp"].dt.year
    time_data["DayOfWeek"] = time_data["Timestamp"].dt.dayofweek
    time_data["DayOfYear"] = time_data["Timestamp"].dt.dayofyear
    time_data["IsWeekend"] = time_data["DayOfWeek"].isin([5, 6]).astype(bool)
    # Get the holidays for the country
    import holidays

    country_holidays_obj = holidays.CountryHoliday(country_holidays)
    time_data["IsHoliday"] = time_data["Timestamp"].dt.date.isin(country_holidays_obj).astype(bool)
    working_days = (time_data["IsWeekend"] == 0) & (time_data["IsHoliday"] == 0)
    time_data["IsWorkingDay"] = working_days.astype(bool)
    return time_data


# Transform


def transform_time_data(time_data: pd.DataFrame) -> pd.DataFrame:
    """Apply cyclic sine/cosine encoding to all numeric time-feature columns.

    Boolean columns are left unchanged; all other numeric columns are replaced with
    ``<col>_sin`` and ``<col>_cos`` features.

    Args:
        time_data: DataFrame produced by :func:`generate_time_data`.

    Returns:
        Transformed DataFrame with cyclic-encoded columns in place of the originals.

    """
    # All bool remain as bool, no transformation needed
    # All other columns sin cos transformation
    for col in time_data.columns:
        if col == "Timestamp" or time_data[col].dtype == "bool":
            continue
        else:
            max_val = time_data[col].max()
            time_data[f"{col}_sin"] = np.sin(2 * np.pi * time_data[col] / max_val)
            time_data[f"{col}_cos"] = np.cos(2 * np.pi * time_data[col] / max_val)
            time_data.drop(columns=[col], inplace=True)

    return time_data


def transform_weather_data(weather_data: pd.DataFrame) -> pd.DataFrame:
    """Normalise numeric weather columns to [0, 1] and resample to hourly frequency.

    Args:
        weather_data: Raw weather DataFrame with a ``Timestamp`` column.

    Returns:
        Normalised, hourly-resampled weather DataFrame.

    """
    # Normalize numeric columns except Timestamp
    numeric_cols = weather_data.select_dtypes(include="number").columns
    for col in numeric_cols:
        if col == "Timestamp":
            continue
        else:
            col_min = weather_data[col].min()
            col_max = weather_data[col].max()
            if col_max != col_min:
                weather_data[col] = (weather_data[col] - col_min) / (col_max - col_min)
            else:
                weather_data[col] = 0.0
    # Make the data hourly by forward filling the values for each hour
    weather_data = weather_data.set_index("Timestamp").resample("h").ffill().reset_index()

    return weather_data


def transform_sun_data(sun_data: pd.DataFrame) -> pd.DataFrame:
    """Compute an hourly solar intensity feature from daily sun data.

    Resamples daily sunrise/sunset/daylight data to hourly resolution and derives
    a ``Solar_intensity`` column using a cosine model centred on solar noon.

    Args:
        sun_data: Daily DataFrame with ``Timestamp``, ``sunrise``, ``sunset`` and
            ``daylight_duration`` columns.

    Returns:
        Hourly DataFrame with a ``Solar_intensity`` column; sunrise/sunset/daylight
        columns are dropped.

    """
    # Transform to hourly data by forward filling the values for each hour
    sun_data = sun_data.set_index("Timestamp").resample("h").ffill().reset_index()
    sunrise = pd.to_datetime(sun_data["sunrise"], errors="coerce").dt.tz_localize(None)
    day_length_hours = sun_data["daylight_duration"] / 3600.0
    safe_day_length = day_length_hours.replace(0, np.nan)
    solar_noon_dt = sunrise + pd.to_timedelta(day_length_hours / 2, unit="h")
    hours = (sun_data["Timestamp"] - solar_noon_dt) / np.timedelta64(1, "h")
    sun_data["Solar_intensity"] = np.maximum(0, np.cos(hours * np.pi / safe_day_length)).fillna(0.0)
    sun_data.drop(columns=["sunrise", "sunset", "daylight_duration"], inplace=True)

    return sun_data


def transform_price_data(price_data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate price data to hourly resolution and drop redundant time columns.

    Args:
        price_data: Raw price DataFrame with a ``Timestamp`` column.

    Returns:
        Hourly-aggregated price DataFrame.

    """
    columns_to_drop = [
        col for col in ["hour", "dayofweek", "month", "is_weekend"] if col in price_data.columns
    ]
    if columns_to_drop:
        price_data = price_data.drop(columns=columns_to_drop)

    numeric_cols = price_data.select_dtypes(include="number").columns
    non_numeric_cols = [
        col for col in price_data.columns if col not in numeric_cols and col != "Timestamp"
    ]
    aggregated = (
        price_data.set_index("Timestamp")
        .resample("h")
        .agg({**dict.fromkeys(numeric_cols, "mean"), **dict.fromkeys(non_numeric_cols, "first")})
    )
    return aggregated.reset_index()


def get_timestamp_range(dataframe: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the minimum and maximum timestamp values in a DataFrame.

    Args:
        dataframe: DataFrame with a ``Timestamp`` column.

    Returns:
        Tuple of ``(min_timestamp, max_timestamp)``.

    """
    timestamps = pd.to_datetime(dataframe["Timestamp"], errors="coerce")
    return timestamps.min(), timestamps.max()


def filter_by_timestamp_range(
    dataframe: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Filter a DataFrame to rows whose ``Timestamp`` falls within [start_date, end_date].

    Args:
        dataframe: DataFrame with a ``Timestamp`` column.
        start_date: Inclusive lower bound for the timestamp filter.
        end_date: Inclusive upper bound for the timestamp filter.

    Returns:
        Filtered copy of the DataFrame.

    """
    mask = (dataframe["Timestamp"] >= start_date) & (dataframe["Timestamp"] <= end_date)
    return dataframe.loc[mask].copy()


def log_dataset_ranges(ranges_by_name: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> None:
    """Print the timestamp range for each dataset to stdout.

    Args:
        ranges_by_name: Mapping from dataset name to ``(min_timestamp, max_timestamp)``.

    """
    print("Timestamp ranges used for merge:")
    for name, (start_date, end_date) in ranges_by_name.items():
        print(f"- {name}: {start_date} to {end_date}")


def merge_datasets(
    price_data: pd.DataFrame,
    weather_data: pd.DataFrame,
    sun_data: pd.DataFrame,
    time_data: pd.DataFrame,
) -> pd.DataFrame:
    """Merge price, weather, sun and time DataFrames on the ``Timestamp`` column.

    Args:
        price_data: Hourly price DataFrame.
        weather_data: Hourly weather DataFrame.
        sun_data: Hourly solar intensity DataFrame.
        time_data: Hourly time-feature DataFrame.

    Returns:
        Left-merged DataFrame combining all four sources on ``Timestamp``.

    """
    # Merge on Timestamp
    merged_data = price_data.merge(weather_data, on="Timestamp", how="left")
    merged_data = merged_data.merge(sun_data, on="Timestamp", how="left")
    merged_data = merged_data.merge(time_data, on="Timestamp", how="left")

    return merged_data


def add_lagged_features(
    dataframe: pd.DataFrame,
    lag_steps: tuple[int, ...],
    drop_na: bool = True,
) -> pd.DataFrame:
    """Append lagged copies of all numeric columns to a DataFrame.

    Args:
        dataframe: Time-sorted DataFrame with a ``Timestamp`` column.
        lag_steps: Sequence of integer lag offsets (in rows) to generate.
        drop_na: If ``True``, rows introduced as ``NaN`` by the maximum lag are dropped.

    Returns:
        DataFrame extended with ``<col>_lag_<n>h`` columns for every numeric column
        and each lag step.

    Raises:
        ValueError: If the DataFrame does not contain a ``Timestamp`` column.

    """
    if "Timestamp" not in dataframe.columns:
        raise ValueError("Timestamp column missing; cannot build lagged features.")

    dataframe = dataframe.sort_values("Timestamp").reset_index(drop=True)

    numeric_cols = dataframe.select_dtypes(include=["number", "bool"]).columns
    numeric_cols = [col for col in numeric_cols if col != "Timestamp"]

    for col in numeric_cols:
        for step in lag_steps:
            dataframe[f"{col}_lag_{step}h"] = dataframe[col].shift(step)

    if drop_na:
        max_lag = max(lag_steps) if lag_steps else 0
        if max_lag > 0:
            dataframe = dataframe.iloc[max_lag:].reset_index(drop=True)

    return dataframe


def preprocess_merge(
    price_data: pd.DataFrame,
    weather_data: pd.DataFrame,
    sun_data: pd.DataFrame,
    lag_steps: tuple[int, ...] = (1, 2, 3, 6, 12, 24),
    drop_na: bool = True,
) -> pd.DataFrame:
    """Run the full ETL pipeline: align, transform, merge and add lagged features.

    Determines the overlapping timestamp range across all three sources, applies
    per-source transformations, merges them, then engineers lagged features.

    Args:
        price_data: Raw price DataFrame with a ``Timestamp`` column.
        weather_data: Raw hourly weather DataFrame.
        sun_data: Raw daily sun DataFrame.
        lag_steps: Lag offsets (in hours) to create for all numeric features.
        drop_na: Whether to drop rows with ``NaN`` values introduced by lagging.

    Returns:
        Fully processed and merged DataFrame ready for model training.

    Raises:
        ValueError: If any dataset has an invalid timestamp range or there is no
            overlapping time window across all three sources.

    """
    price_range = get_timestamp_range(price_data)
    weather_range = get_timestamp_range(weather_data)
    sun_range = get_timestamp_range(sun_data)
    log_dataset_ranges(
        {
            "price_data": price_range,
            "historical_hourly": weather_range,
            "historical_daily": sun_range,
        }
    )

    start_date = max(price_range[0], weather_range[0], sun_range[0])
    end_date = min(price_range[1], weather_range[1], sun_range[1])
    if pd.isna(start_date) or pd.isna(end_date):
        raise ValueError(
            "Invalid timestamp range detected; check input data for missing timestamps."
        )
    if start_date > end_date:
        raise ValueError("No overlapping timestamp range across datasets.")

    price_data = filter_by_timestamp_range(price_data, start_date, end_date)
    weather_data = filter_by_timestamp_range(weather_data, start_date, end_date)
    sun_data = filter_by_timestamp_range(sun_data, start_date, end_date)

    time_data = generate_time_data(start_date=start_date, end_date=end_date, country_holidays="UK")
    time_data = transform_time_data(time_data)
    weather_data = transform_weather_data(weather_data)
    sun_data = transform_sun_data(sun_data)
    price_data = transform_price_data(price_data)
    merged_data = merge_datasets(price_data, weather_data, sun_data, time_data)
    merged_data = add_lagged_features(merged_data, lag_steps=lag_steps, drop_na=drop_na)
    return merged_data


def main() -> None:
    """Run the ETL pipeline from the default file layout and save the merged dataset."""
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / ".." / "Data"
    historical_daily_path = data_dir / "historical_daily_2025.csv"
    historical_hourly_path = data_dir / "historical_hourly_2025.csv"
    price_data_path = data_dir / "price_data.csv"

    price_data = pd.read_csv(price_data_path)
    weather_data = pd.read_csv(historical_hourly_path)
    sun_data = pd.read_csv(historical_daily_path)

    price_data = standardize_timestamp_column(price_data, ["timestamp", "ts_utc", "Timestamp"])
    weather_data = standardize_timestamp_column(weather_data, ["timestamp_utc", "Timestamp"])
    sun_data = standardize_timestamp_column(sun_data, ["date_utc", "Timestamp"])

    expected_columns_price_data = price_data.columns.tolist()
    expected_columns_weather = weather_data.columns.tolist()
    expected_columns_sun = sun_data.columns.tolist()

    # Save expected columns to a .txt file
    with open("expected_columns.txt", "w") as f:
        f.write("Expected columns for price_data:\n")
        f.write(", ".join(expected_columns_price_data) + "\n\n")

        f.write("Expected columns for historical_hourly:\n")
        f.write(", ".join(expected_columns_weather) + "\n\n")

        f.write("Expected columns for historical_daily:\n")
        f.write(", ".join(expected_columns_sun) + "\n")

    merged = preprocess_merge(price_data, weather_data, sun_data)

    # Save the merged dataset to a new csv file
    merged.to_csv(base_dir / ".." / "Data" / "merged_dataset.csv", index=False)


if __name__ == "__main__":
    main()
