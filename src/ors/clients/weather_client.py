"""Client utilities for fetching weather data from the Open-Meteo API."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import openmeteo_requests
import pandas as pd
import requests_cache
from openmeteo_requests import Client
from openmeteo_sdk.WeatherApiResponse import WeatherApiResponse  # type: ignore[import-untyped]
from retry_requests import retry

from ors.config.api_endpoints import ARCHIVE_API_URL, FORECAST_API_URL

DEFAULT_PARAMS = {
    "latitude": 54.727592,
    "longitude": -2.6679993,
    "hourly": [
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
        "is_day",
        "shortwave_radiation",
        "direct_radiation",
        "wind_speed_10m",
        "wind_gusts_10m",
        "wind_direction_10m",
        "surface_pressure",
        "weather_code",
    ],
    "daily": [
        "weather_code",
        "sunrise",
        "sunset",
        "daylight_duration",
        "shortwave_radiation_sum",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "precipitation_hours",
        "temperature_2m_max",
        "temperature_2m_min",
    ],
    "current": [
        "is_day",
        "temperature_2m",
        "precipitation",
        "rain",
        "cloud_cover",
        "cloud_cover_low",
        "wind_speed_10m",
    ],
    "models": "ukmo_uk_deterministic_2km",
    "timezone": "GMT",
    "forecast_days": 1,
    "forecast_hours": 24,
    "past_hours": 6,
}


class WeatherFetcherError(Exception):
    """Raised when weather data fetching or formatting fails in a controlled way."""


def make_client() -> Client | None:
    """Create an Open-Meteo API client with caching and retry support.

    Returns:
        Configured openmeteo_requests Client instance.

    """
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_forecast(client: Client, params: dict[str, Any] | None = None) -> Any:
    """Fetch the current/hourly/daily forecast and return the first API response object.

    Args:
        client: Open-Meteo client created by :func:`make_client`.
        params: API request parameters; defaults to :data:`DEFAULT_PARAMS` when ``None``.

    Returns:
        The first Open-Meteo response object returned by the API.

    Raises:
        WeatherFetcherError: If the API returns an empty response list.

    """
    if params is None:
        params = DEFAULT_PARAMS

    responses = client.weather_api(FORECAST_API_URL, params=params)
    if not responses:
        raise WeatherFetcherError("No responses returned from Open-Meteo Forecast API.")
    return responses[0]


def fetch_hist_hourly(
    client: Client,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    hourly_vars: list[str],
) -> pd.DataFrame:
    """Fetch historical hourly data from the Open-Meteo Archive API.

    Args:
        client: Open-Meteo client created by :func:`make_client`.
        latitude: Latitude of the target location in decimal degrees.
        longitude: Longitude of the target location in decimal degrees.
        start_date: Inclusive start date in ``YYYY-MM-DD`` format.
        end_date: Inclusive end date in ``YYYY-MM-DD`` format.
        hourly_vars: List of Open-Meteo hourly variable names to request.

    Returns:
        DataFrame with a ``timestamp_utc`` column plus one column per requested variable.

    Raises:
        WeatherFetcherError: If the API returns an empty response list.

    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": hourly_vars,
        "timezone": "GMT",
    }

    responses = client.weather_api(ARCHIVE_API_URL, params=params)
    if not responses:
        raise WeatherFetcherError(
            "No historical hourly responses returned from Open-Meteo Archive API."
        )

    return to_hourly_df(responses[0], hourly_vars)


def fetch_hist_daily(
    client: Client,
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    daily_vars: list[str],
) -> pd.DataFrame:
    """Fetch historical daily data from the Open-Meteo Archive API.

    Args:
        client: Open-Meteo client created by :func:`make_client`.
        latitude: Latitude of the target location in decimal degrees.
        longitude: Longitude of the target location in decimal degrees.
        start_date: Inclusive start date in ``YYYY-MM-DD`` format.
        end_date: Inclusive end date in ``YYYY-MM-DD`` format.
        daily_vars: List of Open-Meteo daily variable names to request.

    Returns:
        DataFrame with a ``date_utc`` column plus one column per requested variable.

    Raises:
        WeatherFetcherError: If the API returns an empty response list.

    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": daily_vars,
        "timezone": "GMT",
    }

    responses = client.weather_api(ARCHIVE_API_URL, params=params)
    if not responses:
        raise WeatherFetcherError(
            "No historical daily responses returned from Open-Meteo Archive API."
        )

    return to_daily_df(responses[0], daily_vars)


def to_hourly_df(api_response: WeatherApiResponse, hourly_vars: list[str]) -> pd.DataFrame:
    """Build an hourly forecast or history API response into a clean DataFrame.

    Args:
        api_response: Open-Meteo response object containing an hourly data block.
        hourly_vars: Ordered list of variable names corresponding to the response variables.

    Returns:
        DataFrame with a ``timestamp_utc`` column and one column per variable.

    Raises:
        WeatherFetcherError: If the response has no hourly block or a variable is missing.

    """
    # Real Open-Meteo response has Hourly()
    if hasattr(api_response, "Hourly"):
        hourly_block = api_response.Hourly()

    # Unit-test fake response has hourly() (method) OR hourly (already an object)
    elif hasattr(api_response, "hourly"):
        hourly_attr = api_response.hourly
        hourly_block = hourly_attr() if callable(hourly_attr) else hourly_attr

    else:
        raise WeatherFetcherError("API response has no hourly block (Hourly() or hourly()/hourly).")

    timestamps = pd.date_range(
        start=pd.to_datetime(hourly_block.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly_block.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly_block.Interval()),
        inclusive="left",
    )

    data = {"timestamp_utc": timestamps}

    for i, var_name in enumerate(hourly_vars):
        try:
            values = hourly_block.Variables(i).ValuesAsNumpy()
        except Exception as e:
            raise WeatherFetcherError(f"Missing hourly variable index {i} for '{var_name}'") from e

        if len(values) != len(timestamps):
            raise WeatherFetcherError(
                f"Length mismatch for {var_name}: "
                f"{len(values)} values vs {len(timestamps)} timestamps"
            )

        data[var_name] = values

    return pd.DataFrame(data=data)


def solar_radiance_15_mins(
    client: Client,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
) -> pd.DataFrame:
    """Fetch 15-minute solar radiance data for PV power generation modeling.

    Requests data from the Open-Meteo Forecast API and trims the response to
    the ``[start_datetime, end_datetime]`` window.  The API returns data from
    the start of the day, so rows before *start_datetime* are discarded.  Rows
    after *end_datetime* are likewise discarded.

    If the value at *start_datetime* is null (i.e. too far in the future for
    the model to have a prediction), a :class:`WeatherFetcherError` is raised.
    Any other null values between start and end are silently dropped.

    Args:
        client: Open-Meteo client created by :func:`make_client`.
        start_datetime: Timezone-aware start of the requested window (inclusive).
            Defaults to the current UTC time.
        end_datetime: Timezone-aware end of the requested window (inclusive).
            Defaults to 48 hours after *start_datetime*.

    Returns:
        DataFrame with ``timestamp_utc`` and ``shortwave_radiation`` columns
        at 15-minute intervals, containing only rows with valid radiation data
        within the requested time window.

    Raises:
        WeatherFetcherError: If the API returns an empty response, or if the
            value at *start_datetime* is null (start is too far in the future).

    """
    if start_datetime is None:
        start_datetime = datetime.now(tz=timezone.utc)
    if end_datetime is None:
        end_datetime = start_datetime + timedelta(hours=48)

    if start_datetime > end_datetime:
        raise WeatherFetcherError(
            f"start_datetime ({start_datetime}) must not be after end_datetime ({end_datetime})"
        )

    # Calculate the number of 15-minute timesteps from the start of the
    # start day until the end datetime (the API returns data from midnight).
    start_of_day = start_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    total_seconds = (end_datetime - start_of_day).total_seconds()
    timesteps = int(total_seconds / (15 * 60)) + 1

    params: dict[str, Any] = {
        "latitude": DEFAULT_PARAMS["latitude"],
        "longitude": DEFAULT_PARAMS["longitude"],
        "minutely_15": ["shortwave_radiation"],
        "forecast_minutely_15": timesteps,
        "models": "ukmo_uk_deterministic_2km",
        "timezone": "GMT",
    }

    try:
        responses = client.weather_api(FORECAST_API_URL, params=params)
    except Exception as e:
        raise WeatherFetcherError(
            f"Open-Meteo API request failed for solar radiance "
            f"(start={start_datetime}, end={end_datetime}, timesteps={timesteps}): {e}"
        ) from e

    if not responses:
        raise WeatherFetcherError(
            "No responses returned from Open-Meteo Forecast API for solar radiance."
        )

    api_response = responses[0]
    minutely_block = api_response.Minutely15()

    timestamps = pd.date_range(
        start=pd.to_datetime(minutely_block.Time(), unit="s", utc=True),
        end=pd.to_datetime(minutely_block.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(minutes=15),
        inclusive="left",
    )

    data = {
        "timestamp_utc": timestamps,
        "shortwave_radiation": pd.to_numeric(
            minutely_block.Variables(0).ValuesAsNumpy(),
            errors="coerce",
        ),
    }

    df = pd.DataFrame(data=data)

    # Trim some rows may be outside the requested window, so filter to the window
    df = df[(df["timestamp_utc"] >= start_datetime) & (df["timestamp_utc"] <= end_datetime)]

    if df.empty:
        raise WeatherFetcherError(
            "No data available for the requested time window "
            f"[{start_datetime}, {end_datetime}]."
        )

    # Check that the first row (at start_datetime) has a valid value
    first_value = df.iloc[0]["shortwave_radiation"]
    if pd.isna(first_value):
        raise WeatherFetcherError(
            f"No forecast data available at start_datetime ({start_datetime}). "
            "The requested time may be too far in the future."
        )

    # Drop any remaining rows with null radiation values
    df = df.dropna(subset=["shortwave_radiation"]).reset_index(drop=True)

    return df


def to_daily_df(api_response: WeatherApiResponse, daily_vars: list[str]) -> pd.DataFrame:
    """Build a daily forecast or history API response into a clean DataFrame.

    Args:
        api_response: Open-Meteo response object containing a daily data block.
        daily_vars: Ordered list of variable names corresponding to the response variables.

    Returns:
        DataFrame with a ``date_utc`` column and one column per variable.

    """
    daily_block = api_response.Daily()

    dates = pd.date_range(
        start=pd.to_datetime(daily_block.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily_block.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily_block.Interval()),
        inclusive="left",
    )

    data = {"date_utc": dates}

    for i, var_name in enumerate(daily_vars):
        if var_name in ("sunrise", "sunset"):
            data[var_name] = pd.to_datetime(
                daily_block.Variables(i).ValuesInt64AsNumpy(),
                unit="s",
                utc=True,
            )
        else:
            data[var_name] = daily_block.Variables(i).ValuesAsNumpy()

    return pd.DataFrame(data=data)


def to_current(api_response: WeatherApiResponse, current_vars: list[str]) -> dict[str, Any]:
    """Build current-conditions data from an API response into a plain dictionary.

    Args:
        api_response: Open-Meteo response object containing a current data block.
        current_vars: Ordered list of variable names corresponding to the current variables.

    Returns:
        Dictionary with ``time_unix``, ``time_utc``, and one entry per variable.

    """
    current_block = api_response.Current()

    out: dict[str, Any] = {}
    time_unix = int(current_block.Time())
    out["time_unix"] = time_unix
    out["time_utc"] = pd.to_datetime(time_unix, unit="s", utc=True)

    for i, var_name in enumerate(current_vars):
        out[var_name] = current_block.Variables(i).Value()

    return out
