"""Connect Weather API outputs to the PV model.

This module provides a glue layer between the Weather API client
(Open-Meteo) and the PV domain model.

Goal:
- Fetch hourly Open-Meteo weather via existing client code.
- Transform weather data into PVTelemetry.
- Feed telemetry into update_pv_state to produce PVState values.
- Provide date-range power generation from a PVSiteConfig.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast

import pandas as pd

from ors.clients import weather_client
from ors.config.pv_config import PVSiteConfig
from ors.domain.models.pv import PVSpec, PVState, PVTelemetry
from ors.services.pv_status import update_pv_state
from ors.utils.pv_converter import pv_site_config_to_spec


@dataclass(frozen=True)
class WeatherToPVColumns:
    """Column mapping for an Open-Meteo hourly DataFrame."""

    timestamp_utc: str = "timestamp_utc"
    shortwave_radiation_w_per_m2: str = "shortwave_radiation"


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Ensure required columns exist in the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to check
        required (Iterable[str]): Column names that must be present

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. " f"Available: {list(df.columns)}")


def hourly_weather_df_to_pv_telemetry(
    hourly_df: pd.DataFrame,
    *,
    cols: WeatherToPVColumns = WeatherToPVColumns(),
) -> list[PVTelemetry]:
    """Convert an Open-Meteo hourly DataFrame into PVTelemetry objects.

    Expected input (from weather_client.to_hourly_df):
        - timestamp_utc: timezone-aware timestamps
        - shortwave_radiation: W/m^2

    PVTelemetry expects solar_radiance_kw_per_m2, so conversion is:
        kW/m^2 = (W/m^2) / 1000.

    Args:
        hourly_df (pd.DataFrame): Open-Meteo hourly weather data
        cols (WeatherToPVColumns): Column mapping configuration

    Returns:
        list[PVTelemetry]: List of telemetry objects with solar radiance data
    """
    _require_columns(
        hourly_df,
        [cols.timestamp_utc, cols.shortwave_radiation_w_per_m2],
    )

    ts = pd.to_datetime(
        hourly_df[cols.timestamp_utc],
        utc=True,
        errors="raise",
    )

    sw_w = pd.to_numeric(
        hourly_df[cols.shortwave_radiation_w_per_m2],
        errors="coerce",
    )

    telemetry: list[PVTelemetry] = []

    for t, sw in zip(ts, sw_w, strict=False):
        radiance_kw_m2: float | None = None

        if pd.notna(sw):
            radiance_kw_m2 = float(sw) / 1000.0

        telemetry.append(
            PVTelemetry(
                timestamp=t.to_pydatetime(),
                generation_kw=None,
                solar_radiance_kw_per_m2=radiance_kw_m2,
            )
        )

    return telemetry


def pv_states_from_hourly_weather_df(
    spec: PVSpec,
    hourly_df: pd.DataFrame,
    *,
    timestep_minutes: int = 60,
    cols: WeatherToPVColumns = WeatherToPVColumns(),
) -> list[PVState]:
    """Generate PVState values from an hourly weather DataFrame.

    This function converts hourly Open-Meteo data into PVTelemetry,
    then applies the PV model logic to compute PVState outputs.

    Args:
        spec (PVSpec): PV system specification
        hourly_df (pd.DataFrame): Hourly weather data
        timestep_minutes (int): Time step duration in minutes
        cols (WeatherToPVColumns): Column mapping configuration

    Returns:
        list[PVState]: List of computed PV states
    """
    telemetry = hourly_weather_df_to_pv_telemetry(
        hourly_df,
        cols=cols,
    )

    return [
        update_pv_state(
            spec,
            t,
            timestep_minutes=timestep_minutes,
        )
        for t in telemetry
    ]


def fetch_live_pv_states_from_openmeteo(
    spec: PVSpec,
    *,
    client: Any | None = None,
    params: dict[str, Any] | None = None,
    timestep_minutes: int = 60,
) -> list[PVState]:
    """Fetch live Open-Meteo forecast and return PVState values.

    This is an end-to-end integration:
        1. Fetch forecast from Open-Meteo.
        2. Convert to hourly DataFrame.
        3. Transform to PVTelemetry.
        4. Produce PVState outputs via PV model logic.

    Args:
        spec (PVSpec): PV system specification
        client (Any | None): Open-Meteo client instance
        params (dict[str, Any] | None): API request parameters
        timestep_minutes (int): Time step duration in minutes

    Returns:
        list[PVState]: List of computed PV states from live forecast
    """
    if client is None:
        client = weather_client.make_client()

    if params is None:
        params = weather_client.DEFAULT_PARAMS

    api_resp = weather_client.fetch_forecast(client, params)  # type: ignore[arg-type]

    hourly_vars = cast(list[str], params["hourly"])
    hourly_df = weather_client.to_hourly_df(
        api_resp,
        hourly_vars,
    )

    return pv_states_from_hourly_weather_df(
        spec,
        hourly_df,
        timestep_minutes=timestep_minutes,
    )


def generate_pv_power_for_date_range(
    config: PVSiteConfig,
    *,  # this forces all following args to be passed as keywords, improving readability and reducing errors
    client: Any | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
) -> pd.DataFrame:
    """Generate 15-minute PV power output for a datetime range.

    Fetches 15-minute solar irradiance forecasts from the Open-Meteo API,
    converts the site configuration to a domain specification, and applies
    PV power calculation logic to produce generation estimates.

    Args:
        config (PVSiteConfig): PV site configuration (MW units). Use a predefined config
            from :data:`ors.config.pv_config.PV_SITE_CONFIGS` or create a
            custom :class:`~ors.config.pv_config.PVSiteConfig`.
        client (Any | None): Open-Meteo client; created automatically when ``None``.
        start_datetime (datetime | None): Timezone-aware start of the requested window.
            Defaults to the current UTC time.
        end_datetime (datetime | None): Timezone-aware end of the requested window.
            Defaults to 48 hours after *start_datetime*.

    Returns:
        pd.DataFrame: DataFrame with ``timestamp_utc`` and ``generation_kw`` columns
            at 15-minute intervals.

    Raises:
        WeatherFetcherError: If the weather API call fails or returns
            no data for the requested window.

    """
    if client is None:
        client = weather_client.make_client()

    solar_df = weather_client.solar_radiance_15_mins(
        client,  # type: ignore[arg-type]
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    spec: PVSpec = pv_site_config_to_spec(config)

    states = pv_states_from_hourly_weather_df(
        spec,
        solar_df,
        timestep_minutes=15,
    )

    return pd.DataFrame(
        {
            "timestamp_utc": [s.timestamp for s in states],
            "generation_kw": [s.generation_kw for s in states],
        }
    )
