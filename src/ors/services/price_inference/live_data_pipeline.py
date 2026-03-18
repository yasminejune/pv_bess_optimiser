"""Live data pipeline: fetch and merge real-time weather and price data.

Fetches Weather and Price inputs from the respective APIs.

Runs and identical ETL transformation process used offline so that the resulting DataFrame is
feature-compatible with the trained model.

"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from typing import cast

import numpy as np
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
from ors.services.prediction.data_pipeline import preprocess_raw_data
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

# Must be >= the largest lag step used during training (XGBoost path)
PRICE_LOOKBACK_HOURS: int = 48

# Must cover the full window_size of the LGBM ForecasterRecursive (max_lag=672
# steps × 15 min = 168 hours).  Add a small buffer to survive minor API gaps.
LGBM_PRICE_LOOKBACK_HOURS: int = 170

# BMRS MID dataset enforces a strict 7-day (168-hour) From→To window.
_MID_MAX_WINDOW_HOURS: int = 168

# Helpers


def _chunked_fetch(
    fetch_fn: Callable[[datetime, datetime], pd.DataFrame],
    start_utc: datetime,
    end_utc: datetime,
    max_hours: int = _MID_MAX_WINDOW_HOURS,
) -> pd.DataFrame:
    """Call fetch_fn in ≤max_hours windows and return concatenated, deduplicated results.

    Some BMRS endpoints (e.g. MID) reject requests whose From→To span exceeds
    7 days.  This helper transparently slices a wider window into compliant
    chunks and stitches the results together.

    Args:
        fetch_fn: Callable ``(start_utc, end_utc) -> pd.DataFrame``.
        start_utc: Start of the full desired window (UTC-aware).
        end_utc: End of the full desired window (UTC-aware).
        max_hours: Maximum window size per individual API call (default 168 h).

    Returns:
        Concatenated DataFrame, deduplicated on ``ts_utc`` (last value wins).
    """
    frames: list[pd.DataFrame] = []
    delta = timedelta(hours=max_hours)
    chunk_start = start_utc
    while chunk_start < end_utc:
        chunk_end = min(chunk_start + delta, end_utc)
        df = fetch_fn(chunk_start, chunk_end)
        if not df.empty:
            frames.append(df)
        chunk_start = chunk_end
    if not frames:
        return fetch_fn(start_utc, start_utc)
    combined = pd.concat(frames, ignore_index=True)
    if "ts_utc" in combined.columns:
        combined = (
            combined.drop_duplicates(subset=["ts_utc"], keep="last")
            .sort_values("ts_utc")
            .reset_index(drop=True)
        )
    return combined


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

    mid = _chunked_fetch(
        lambda s, e: fetch_mid_price(session, base_url, s, e, data_providers, timeout),
        start_utc,
        now,
    )
    itsdo = _chunked_fetch(
        lambda s, e: fetch_itsdo_demand(session, base_url, s, e, timeout=timeout),
        start_utc,
        now,
    )
    indo = _chunked_fetch(
        lambda s, e: fetch_indo_initial_demand(session, base_url, s, e, timeout=timeout),
        start_utc,
        now,
    )
    inddem = _chunked_fetch(
        lambda s, e: fetch_inddem_demand(session, base_url, s, e, timeout=timeout),
        start_utc,
        now,
    )

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


def build_live_lgbm_dataset(
    forecast_steps: int = 96,
    past_hours: int = LGBM_PRICE_LOOKBACK_HOURS,
    base_url: str = BMRS_BASE_URL,
    data_providers: list[str] = DEFAULT_DATA_PROVIDERS,
    timeout: int = 60,
    reference_time: datetime | None = None,
) -> pd.DataFrame:
    """Fetch and preprocess live data for LGBM ForecasterRecursive inference.

    Applies the exact same preprocessing pipeline as the training
    :func:`~ors.services.prediction.data_pipeline.build_merged_dataset` so
    that the resulting feature columns (returned by
    :func:`~ors.services.prediction.data_pipeline.get_feature_cols`) are
    compatible with a fitted ``ForecasterRecursive``.

    The returned DataFrame covers:

    * **Past** (``past_hours`` rows at 15-min resolution): rows where ``price``
      is a real value — used to extract the ``last_window`` for the forecaster.
    * **Future** (``forecast_steps`` rows at 15-min resolution): rows where
      ``price`` is ``NaN`` — used as ``exog`` for
      ``model.predict(steps, last_window, exog)``.

    Args:
        forecast_steps: Number of 15-min steps to include in the future exog
            window (default 96 = 24 h).
        past_hours: Hours of price/weather history to fetch.  Must be >= the
            model's ``window_size`` (168 h for the default LGBM config).
        base_url: BMRS API base URL.
        data_providers: MID data-provider filter list (empty = no filter).
        timeout: HTTP timeout in seconds.
        reference_time: UTC datetime to treat as "now".  If ``None``, runs live.

    Returns:
        Merged, preprocessed DataFrame sorted by Timestamp.  Past rows have a
        valid ``price`` column; future rows have ``price = NaN``.  All exog
        feature columns are populated for both past and future rows.

    Raises:
        ValueError: If the weather API returns no usable data.
    """
    now = reference_time if reference_time is not None else datetime.now(timezone.utc)
    now_naive = (
        pd.Timestamp(now).tz_convert(None)
        if pd.Timestamp(now).tzinfo is not None
        else pd.Timestamp(now)
    )

    # Forecast window: enough days to cover forecast_steps × 15 min ahead,
    # rounded up to whole days with one extra day of buffer.
    forecast_days = max(3, (forecast_steps // (24 * 4)) + 2)

    # 1. Fetch live weather (past + forecast)
    print("Fetching live weather data (LGBM pipeline) ...")
    hourly_df, daily_df = _fetch_live_weather(
        past_hours=past_hours,
        forecast_days=forecast_days,
        reference_time=reference_time,
    )

    # 2. Fetch live price data (past only; future rows will have NaN price)
    print("Fetching live price/demand data ...")
    price_df = _fetch_live_price(
        base_url=base_url,
        data_providers=data_providers,
        lookback_hours=past_hours,
        timeout=timeout,
        reference_time=reference_time,
    )

    if price_df.empty:
        raise ValueError("Price API returned no data — cannot build LGBM inference dataset.")

    # 3. Extend price_df with a single sentinel row at the end of the forecast
    #    window so that preprocess_raw_data builds a full_index that covers
    #    the future exog horizon.
    future_end = now_naive.ceil("15min") + pd.Timedelta(minutes=15 * forecast_steps)
    sentinel = pd.DataFrame({"timestamp": [future_end], "price": [np.nan]})
    for col in price_df.columns:
        if col not in sentinel.columns:
            sentinel[col] = np.nan
    price_df_extended = pd.concat([price_df, sentinel], ignore_index=True)

    # 4. Run the notebook-aligned preprocessing (drop_missing_price=False so
    #    future rows with NaN price are retained in the output).
    print(
        f"Live LGBM pipeline window: {now_naive - pd.Timedelta(hours=past_hours)}  ->  {future_end}"
    )
    df = preprocess_raw_data(
        price_df_extended,
        hourly_df,
        daily_df,
        drop_missing_price=False,
    )

    return df.sort_values("Timestamp").reset_index(drop=True)
