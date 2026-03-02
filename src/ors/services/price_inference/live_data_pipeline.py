"""Live data pipeline: fetch and merge real-time weather and price data.

Fetches Weather and Price inputs from the respective APIs.

Runs and identical ETL transformation process used offline so that the resulting DataFrame is
feature-compatible with the trained model.

"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import cast

import pandas as pd

from ors.clients.weather_client import (
    DEFAULT_PARAMS,
    fetch_forecast,
    make_client,
    to_daily_df,
    to_hourly_df,
)
from ors.etl.etl import (
    add_lagged_features,
    generate_time_data,
    merge_datasets,
    standardize_timestamp_column,
    transform_price_data,
    transform_sun_data,
    transform_time_data,
    transform_weather_data,
)
from ors.services.price_api.price_api import (
    fetch_inddem_demand,
    fetch_indo_initial_demand,
    fetch_itsdo_demand,
    fetch_mid_price,
    make_session,
)

# Constants from APIs

BMRS_BASE_URL: str = "https://data.elexon.co.uk/bmrs/api/v1"
DEFAULT_DATA_PROVIDERS: list[str] = []

# Must be >= the largest lag step used during training
PRICE_LOOKBACK_HOURS: int = 48

# Fetch


def _fetch_live_weather(
    past_hours: int = PRICE_LOOKBACK_HOURS,
    forecast_days: int = 2,
    reference_time: datetime | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch forecast weather using the same DEFAULT_PARAMS as training.

    Args:
        past_hours: Hours of history to include (populates lag windows).
        forecast_days: Days ahead to forecast.
        reference_time: Reserved for future historical-replay support in the
            weather client.

    Returns:
        Tuple of (hourly_df, daily_df) with identical columns to training data.
    """
    params = {
        **DEFAULT_PARAMS,
        "past_hours": past_hours,
        "forecast_days": forecast_days,
        "forecast_hours": forecast_days * 24,
    }
    client = make_client()
    assert client is not None  # make_client always returns a Client
    api_response = fetch_forecast(client, params)
    hourly_vars = cast(list[str], DEFAULT_PARAMS["hourly"])
    daily_vars = cast(list[str], DEFAULT_PARAMS["daily"])
    hourly_df = to_hourly_df(api_response, hourly_vars)
    daily_df = to_daily_df(api_response, daily_vars)
    return hourly_df, daily_df


def _fetch_live_price(
    base_url: str = BMRS_BASE_URL,
    data_providers: list[str] = DEFAULT_DATA_PROVIDERS,
    lookback_hours: int = PRICE_LOOKBACK_HOURS,
    timeout: int = 60,
    reference_time: datetime | None = None,
) -> pd.DataFrame:
    """Fetch recent price and demand data from BMRS using the robust pipeline.

    Args:
        base_url: BMRS API base URL.
        data_providers: Optional MID provider filter list (empty = APXMIDP).
        lookback_hours: How far back to fetch (must cover the longest lag).
        timeout: HTTP request timeout in seconds.
        reference_time: If provided, treat this as "now" for the time window.

    Returns:
        DataFrame with columns: timestamp, price, demand_itsdo, demand_indo,
        demand_inddem.
    """
    session = make_session()
    now = reference_time if reference_time is not None else datetime.now(timezone.utc)
    start_utc = now - timedelta(hours=lookback_hours)

    mid = fetch_mid_price(session, base_url, start_utc, now, data_providers, timeout)
    itsdo = fetch_itsdo_demand(session, base_url, start_utc, now, timeout=timeout)
    indo = fetch_indo_initial_demand(session, base_url, start_utc, now, timeout=timeout)
    inddem = fetch_inddem_demand(session, base_url, start_utc, now, timeout=timeout)

    merged: pd.DataFrame | None = None
    for frame in [mid, itsdo, indo, inddem]:
        if frame.empty:
            continue
        merged = frame if merged is None else merged.merge(frame, on="ts_utc", how="outer")

    if merged is None or merged.empty:
        return pd.DataFrame(
            columns=["timestamp", "price", "demand_itsdo", "demand_indo", "demand_inddem"]
        )

    return (
        merged.rename(columns={"ts_utc": "timestamp"})
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


# Merge the data as expected by the model


def build_live_merged_dataset(
    past_hours: int = PRICE_LOOKBACK_HOURS,
    forecast_days: int = 2,
    lag_steps: tuple[int, ...] = (1, 2, 3, 6, 12, 24),
    base_url: str = BMRS_BASE_URL,
    data_providers: list[str] = DEFAULT_DATA_PROVIDERS,
    timeout: int = 60,
    reference_time: datetime | None = None,
) -> pd.DataFrame:
    """Return a fully ETL-processed DataFrame.

    Applies the identical transformation sequence used during training so that the
    resulting feature columns are compatible with the trained model.

    Args:
        past_hours: Hours of history to fetch; must be >= the largest lag step.
        forecast_days: Days ahead to include from the weather forecast.
        lag_steps: Lag offsets in hours — must match those used during training.
        base_url: BMRS API base URL.
        data_providers: MID data-provider filter list (empty = no filter).
        timeout: HTTP timeout in seconds.
        reference_time: UTC datetime to treat as "now". If None, runs live.

    Returns:
        Merged, transformed DataFrame sorted by Timestamp, ready for inference.

    Raises:
        ValueError: If there is no overlapping timestamp window between weather
            and sun datasets.
    """
    # 1. Fetch raw data
    print("Fetching live weather data ...")
    hourly_df, daily_df = _fetch_live_weather(
        past_hours=past_hours,
        forecast_days=forecast_days,
        reference_time=reference_time,
    )

    print("Fetching live price/demand data ...")
    price_df = _fetch_live_price(
        base_url=base_url,
        data_providers=data_providers,
        lookback_hours=past_hours,
        timeout=timeout,
        reference_time=reference_time,
    )

    # 2. Standardise timestamp columns
    weather_data = standardize_timestamp_column(
        hourly_df, candidates=["timestamp_utc", "Timestamp"]
    )
    sun_data = standardize_timestamp_column(daily_df, candidates=["date_utc", "Timestamp"])
    price_data = standardize_timestamp_column(
        price_df, candidates=["timestamp", "ts_utc", "Timestamp"]
    )

    # 3. Determine the window from weather and sun only.

    weather_ts = pd.to_datetime(weather_data["Timestamp"], errors="coerce")
    sun_ts = pd.to_datetime(sun_data["Timestamp"], errors="coerce")

    start_date = max(weather_ts.min(), sun_ts.min())
    end_date = min(weather_ts.max(), sun_ts.max())

    if pd.isna(start_date) or pd.isna(end_date) or start_date > end_date:
        raise ValueError(
            f"No overlapping timestamp window between weather and sun datasets. "
            f"Weather: {weather_ts.min()} -> {weather_ts.max()}, "
            f"Sun: {sun_ts.min()} -> {sun_ts.max()}"
        )

    print(f"Live pipeline window: {start_date}  ->  {end_date}")

    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        mask = (df["Timestamp"] >= start_date) & (df["Timestamp"] <= end_date)
        return df.loc[mask].copy()

    weather_data = _filter(weather_data)
    sun_data = _filter(sun_data)

    price_data = price_data.loc[price_data["Timestamp"] >= start_date].copy()

    # 4. Generate time features for the full window (past + future)
    time_data = generate_time_data(start_date=start_date, end_date=end_date, country_holidays="UK")

    # 5. Apply per-source ETL transforms (identical to offline pipeline)
    time_data = transform_time_data(time_data)
    weather_data = transform_weather_data(weather_data)
    sun_data = transform_sun_data(sun_data)
    price_data = transform_price_data(price_data)

    # 6. Merge all sources on Timestamp.
    #    Future rows will have NaN for price/demand — expected and correct.
    merged = merge_datasets(price_data, weather_data, sun_data, time_data)

    # 7. Add lagged features
    merged = add_lagged_features(merged, lag_steps=lag_steps, drop_na=False)

    return merged.sort_values("Timestamp").reset_index(drop=True)
