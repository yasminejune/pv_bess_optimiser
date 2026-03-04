from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from ors.config.pv_config import PVSiteConfig, SiteType, get_pv_config
from ors.services.price_inference import run_inference
from ors.services.weather_to_pv import generate_pv_power_for_date_range


def floor_to_prev_15min_utc(dt: datetime) -> datetime:
    """
    Floors a datetime to the previous 15-minute boundary in UTC.
    If datetime is naive, it is assumed to be UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    dt = dt.astimezone(timezone.utc)
    minute = (dt.minute // 15) * 15
    return dt.replace(minute=minute, second=0, microsecond=0)


def create_input_df(
    config: PVSiteConfig = None,
    *,
    client: Any | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Generates a combined DataFrame containing:
        - PV power forecast data
        - Price prediction model output

    Data contract (as implemented):
        - PV dataframe must contain: 'timestamp_utc'
        - Price dataframe must contain: 'Timestamp' (capital T)

    The function:
        1. Determines a time window (default = now rounded to previous 15 min → +1 day).
        2. Generates PV production forecast for that window.
        3. Runs the price prediction model using **kwargs.
        4. Renames price Timestamp -> timestamp_utc.
        5. Merges both datasets on UTC timestamps using an outer merge.

    Parameters
    ----------
    config : PVSiteConfig
        PV site configuration.
    client : Any | None, optional
        Reserved for future use (not used currently).
    start_datetime : datetime | None, optional
        Start of forecasting window. If None, defaults to current UTC time
        floored to previous 15-minute boundary.
    end_datetime : datetime | None, optional
        End of forecasting window. If None, defaults to start_datetime + 1 day.
    **kwargs :
        Forwarded directly to `run_inference`. If `horizon_hours` is not provided,
        it is injected based on (end_datetime - start_datetime).

    Returns
    -------
    pd.DataFrame
        Merged PV + price dataframe, outer-joined on 'timestamp_utc', sorted by time.
    """
    if config == None:
        config = get_pv_config(SiteType.BURST_1)
    use_historic = True
    # --- Resolve start datetime ---
    if start_datetime is None:
        use_historic = False
        start_datetime = floor_to_prev_15min_utc(datetime.now(timezone.utc))
    else:
        start_datetime = floor_to_prev_15min_utc(start_datetime)

    # --- Resolve end datetime ---
    if end_datetime is None:
        end_datetime = start_datetime + timedelta(days=1)
    else:
        if end_datetime.tzinfo is None:
            end_datetime = end_datetime.replace(tzinfo=timezone.utc)
        end_datetime = end_datetime.astimezone(timezone.utc)

    if end_datetime <= start_datetime:
        raise ValueError("end_datetime must be after start_datetime")

    # --- Compute horizon in hours ---
    horizon_hours = int((end_datetime - start_datetime).total_seconds() / 3600.0)

    # --- Generate PV forecast ---
    df_pv = generate_pv_power_for_date_range(
        config=config,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )
    print(df_pv)

    # PV is expected to already have timestamp_utc
    if "timestamp_utc" not in df_pv.columns:
        raise KeyError("'timestamp_utc' column not found in PV dataframe")

    df_pv["timestamp_utc"] = pd.to_datetime(df_pv["timestamp_utc"], utc=True)

    # --- Run price prediction model ---
    if "horizon_hours" not in kwargs:
        kwargs["horizon_hours"] = horizon_hours

    if use_historic:
        df_price = run_inference(reference_time=start_datetime, **kwargs)
    else:
        df_price = run_inference(reference_time=start_datetime, **kwargs)

    # Price is expected to have Timestamp
    if "Timestamp" not in df_price.columns:
        raise KeyError("'Timestamp' column not found in price dataframe")

    df_price = df_price.rename(columns={"Timestamp": "timestamp_utc"})
    df_price["timestamp_utc"] = pd.to_datetime(df_price["timestamp_utc"], utc=True)

    # --- Outer merge ---
    df = (
        df_pv.merge(df_price, on="timestamp_utc", how="outer")
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )

    df.rename(columns={"timestamp_utc": "timestamp", "Price_pred": "price_intraday"}, inplace=True)

    return df[:96]
