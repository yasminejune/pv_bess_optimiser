"""Connect Weather API outputs to the PV model.

This module provides a glue layer between the Weather API client
(Open-Meteo) and the PV domain model.

Goal:
- Fetch hourly Open-Meteo weather via existing client code.
- Transform weather data into PVTelemetry.
- Feed telemetry into update_pv_state to produce PVState values.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from ors.clients import weather_fetcher
from ors.domain.models.pv import PVSpec, PVState, PVTelemetry
from ors.services.pv_status import update_pv_state


@dataclass(frozen=True)
class WeatherToPVColumns:
    """Column mapping for an Open-Meteo hourly DataFrame."""

    timestamp_utc: str = "timestamp_utc"
    shortwave_radiation_w_per_m2: str = "shortwave_radiation"


def _require_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    """Ensure required columns exist in the DataFrame.

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

    Expected input (from weather_fetcher.to_hourly_df):
        - timestamp_utc: timezone-aware timestamps
        - shortwave_radiation: W/m^2

    PVTelemetry expects solar_radiance_kw_per_m2, so conversion is:
        kW/m^2 = (W/m^2) / 1000.
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
    """
    if client is None:
        client = weather_fetcher.make_client()

    if params is None:
        params = weather_fetcher.DEFAULT_PARAMS

    api_resp = weather_fetcher.fetch_forecast(client, params)

    hourly_df = weather_fetcher.to_hourly_df(
        api_resp,
        params["hourly"],
    )

    return pv_states_from_hourly_weather_df(
        spec,
        hourly_df,
        timestep_minutes=timestep_minutes,
    )
