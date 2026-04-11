"""Integration helpers: combine PV forecast and price prediction into a single DataFrame."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ors.config.optimization_config import PVConfiguration
from ors.config.pv_config import PVSiteConfig, SiteType, get_pv_config
from ors.services.price_inference import get_price_forecast_df
from ors.services.price_inference.live_inference import LGBM_MODEL_DIR
from ors.services.weather_to_pv import (
    generate_pv_power_for_date_range,
    generate_runtime_pv_power_for_date_range,
)


def floor_to_prev_15min_utc(dt: datetime) -> datetime:
    """Floor a datetime to the previous 15-minute boundary in UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    dt = dt.astimezone(timezone.utc)
    minute = (dt.minute // 15) * 15
    return dt.replace(minute=minute, second=0, microsecond=0)


def _build_target_index(
    start_datetime: datetime,
    end_datetime: datetime,
    time_step_minutes: int,
) -> pd.DatetimeIndex:
    """Build the canonical timestamp index for the optimization horizon."""
    return pd.date_range(
        start=start_datetime,
        end=end_datetime,
        freq=f"{time_step_minutes}min",
        inclusive="left",
        tz="UTC",
    )


def _resample_pv_to_target(
    pv_df: pd.DataFrame,
    *,
    start_datetime: datetime,
    end_datetime: datetime,
    time_step_minutes: int,
) -> pd.DataFrame:
    """Align PV generation onto the optimization horizon."""
    expected_15 = _build_target_index(start_datetime, end_datetime, 15)
    pv_series = pd.Series(
        pd.to_numeric(pv_df["generation_kw"], errors="coerce").to_numpy(),
        index=pd.to_datetime(pv_df["timestamp_utc"], utc=True),
        dtype="float64",
    ).sort_index()

    aligned_15 = (
        pv_series.reindex(pv_series.index.union(expected_15))
        .sort_index()
        .interpolate(method="time")
        .ffill()
        .bfill()
        .reindex(expected_15)
    )

    target_index = _build_target_index(start_datetime, end_datetime, time_step_minutes)

    if time_step_minutes == 15:
        aligned = aligned_15
    else:
        aligned = aligned_15.resample(
            f"{time_step_minutes}min",
            origin=start_datetime,
        ).mean()

    aligned = aligned.reindex(target_index).fillna(0.0).clip(lower=0.0)

    return pd.DataFrame(
        {
            "timestamp": target_index,
            "generation_kw": aligned.astype(float).to_numpy(),
        }
    )


def _generate_pv_df(
    config: PVSiteConfig | PVConfiguration | None,
    *,
    client: Any | None,
    start_datetime: datetime,
    end_datetime: datetime,
    time_step_minutes: int,
) -> pd.DataFrame:
    """Generate PV output from either a static site config or runtime PV config."""
    if config is None:
        resolved_config: PVSiteConfig | PVConfiguration = get_pv_config(SiteType.BURST_1)
    else:
        resolved_config = config

    if isinstance(resolved_config, PVSiteConfig):
        pv_df = generate_pv_power_for_date_range(
            config=resolved_config,
            client=client,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )
    else:
        if resolved_config.location_lat is None or resolved_config.location_lon is None:
            raise ValueError("PV forecast config requires location_lat and location_lon")
        if resolved_config.panel_area_m2 is None:
            raise ValueError("PV forecast config requires panel_area_m2")

        pv_df = generate_runtime_pv_power_for_date_range(
            latitude=resolved_config.location_lat,
            longitude=resolved_config.location_lon,
            rated_power_kw=resolved_config.rated_power_kw,
            panel_area_m2=resolved_config.panel_area_m2,
            panel_efficiency=resolved_config.panel_efficiency,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            client=client,
            max_export_kw=resolved_config.max_export_kw,
            min_generation_kw=resolved_config.min_generation_kw,
            curtailment_supported=resolved_config.curtailment_supported,
        )

    if "timestamp_utc" not in pv_df.columns:
        raise KeyError("'timestamp_utc' column not found in PV dataframe")

    return _resample_pv_to_target(
        pv_df,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        time_step_minutes=time_step_minutes,
    )


def create_input_df(
    config: PVSiteConfig | PVConfiguration | None = None,
    *,
    client: Any | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
    time_step_minutes: int = 15,
    model_path: Path = LGBM_MODEL_DIR,
    **kwargs: Any,
) -> pd.DataFrame:
    """Generate a combined DataFrame of PV forecast and price prediction output."""
    if start_datetime is None:
        start_datetime = floor_to_prev_15min_utc(datetime.now(timezone.utc))
    else:
        start_datetime = floor_to_prev_15min_utc(start_datetime)

    if end_datetime is None:
        end_datetime = start_datetime + timedelta(days=1)
    else:
        if end_datetime.tzinfo is None:
            end_datetime = end_datetime.replace(tzinfo=timezone.utc)
        end_datetime = end_datetime.astimezone(timezone.utc)

    if end_datetime <= start_datetime:
        raise ValueError("end_datetime must be after start_datetime")

    df_pv = _generate_pv_df(
        config,
        client=client,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        time_step_minutes=time_step_minutes,
    )
    df_price = get_price_forecast_df(
        start_datetime,
        end_datetime,
        time_step_minutes,
        model_path=model_path,
        **kwargs,
    )

    df = df_pv.merge(df_price, on="timestamp", how="left")
    if df["price_intraday"].isna().any():
        raise ValueError("Price forecast did not cover the requested optimization horizon")

    return df.sort_values("timestamp").reset_index(drop=True)
