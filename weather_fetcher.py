import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# API endpoints, global constants
FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"


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
    # Raised when weather data fetching or formatting fails in a controlled way
    pass


def make_client():
    # Create Open-Meteo client with cache + retry
    cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def fetch_forecast(client, params=None):
    # Fetch forecast (current/hourly/daily) and return the first response object
    if params is None:
        params = DEFAULT_PARAMS

    responses = client.weather_api(FORECAST_API_URL, params=params)
    if not responses:
        raise WeatherFetcherError("No responses returned from Open-Meteo Forecast API.")
    return responses[0]


def fetch_hist_hourly(client, latitude, longitude, start_date, end_date, hourly_vars):
    # Fetch historical hourly data from Open-Meteo Archive API and return a DataFrame
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


def fetch_hist_daily(client, latitude, longitude, start_date, end_date, daily_vars):
    # Fetch historical daily data from Open-Meteo Archive API and return a DataFrame
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


def to_hourly_df(api_response, hourly_vars):
    # Build hourly forecast/history into a clean DataFrame

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
                f"Length mismatch for {var_name}: {len(values)} values vs {len(timestamps)} timestamps"
            )

        data[var_name] = values

    return pd.DataFrame(data=data)


def to_daily_df(api_response, daily_vars):
    # Build daily forecast/history into a clean DataFrame
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


def to_current(api_response, current_vars):
    # Build current conditions into a dict
    current_block = api_response.Current()

    out = {}
    time_unix = int(current_block.Time())
    out["time_unix"] = time_unix
    out["time_utc"] = pd.to_datetime(time_unix, unit="s", utc=True)

    for i, var_name in enumerate(current_vars):
        out[var_name] = current_block.Variables(i).Value()

    return out


def main():
    # Run forecast + historical pipeline and save datasets
    client = make_client()

    # Forecast
    api_response = fetch_forecast(client, DEFAULT_PARAMS)

    current = to_current(api_response, DEFAULT_PARAMS["current"])
    hourly_df = to_hourly_df(api_response, DEFAULT_PARAMS["hourly"])
    daily_df = to_daily_df(api_response, DEFAULT_PARAMS["daily"])

    print("Current:")
    for k, v in current.items():
        print(f"  {k}: {v}")

    print("\nForecast hourly rows:", len(hourly_df))
    print("Forecast daily rows:", len(daily_df))

    # Historical: 3 years starting from 2023
    start_date = "2023-01-01"
    end_date = "2025-12-31"

    hist_hourly = fetch_hist_hourly(
        client=client,
        latitude=DEFAULT_PARAMS["latitude"],
        longitude=DEFAULT_PARAMS["longitude"],
        start_date=start_date,
        end_date=end_date,
        hourly_vars=DEFAULT_PARAMS["hourly"],
    )

    hist_daily = fetch_hist_daily(
        client=client,
        latitude=DEFAULT_PARAMS["latitude"],
        longitude=DEFAULT_PARAMS["longitude"],
        start_date=start_date,
        end_date=end_date,
        daily_vars=DEFAULT_PARAMS["daily"],
    )

    # Save files
    hourly_path = "data/historical_hourly_2023_2025.csv"
    daily_path = "data/historical_daily_2023_2025.csv"

    hist_hourly.to_csv(hourly_path, index=False)
    hist_daily.to_csv(daily_path, index=False)

    print(f"\nSaved: {hourly_path}")
    print(f"Saved: {daily_path}")
    print("\nHistorical hourly rows:", len(hist_hourly))
    print("Historical daily rows:", len(hist_daily))


# Backwards-compatible function names expected by tests
build_hourly_dataframe = to_hourly_df

if __name__ == "__main__":
    main()
