import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
open_meteo_client = openmeteo_requests.Client(session=retry_session)

# Make sure all required weather variables are listed here
# The order of variables in hourly or daily is important to assign them correctly below
URL = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 54.7584,
    "longitude": -2.6953,
    "daily": [
        "daylight_duration",
        "shortwave_radiation_sum",
        "sunrise",
        "sunset",
        "weather_code",
        "precipitation_sum",
        "rain_sum",
        "snowfall_sum",
        "precipitation_hours",
        "temperature_2m_max",
        "temperature_2m_min",
    ],
    "hourly": [
        "temperature_2m",
        "direct_radiation",
        "is_day",
        "cloud_cover",
        "precipitation",
        "shortwave_radiation",
        "diffuse_radiation",
        "sunshine_duration",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "relative_humidity_2m",
        "dew_point_2m",
        "wind_speed_10m",
        "wind_gusts_10m",
        "wind_direction_10m",
        "rain",
        "showers",
        "snowfall",
        "surface_pressure",
        "visibility",
        "weather_code",
    ],
    "models": "ukmo_uk_deterministic_2km",
    "current": [
        "is_day",
        "temperature_2m",
        "precipitation",
        "rain",
        "cloud_cover",
        "wind_speed_10m",
    ],
    "timezone": "GMT",
    "forecast_days": 1,
    "forecast_hours": 24,
    "past_hours": 6,
}
api_responses = open_meteo_client.weather_api(URL, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
api_response = api_responses[0]
print(f"Coordinates: {api_response.Latitude()}°N {api_response.Longitude()}°E")
print(f"Elevation: {api_response.Elevation()} m asl")
print(f"Timezone: {api_response.Timezone()}{api_response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0: {api_response.UtcOffsetSeconds()}s")

# Process current data. The order of variables needs to be the same as requested.
current = api_response.Current()
current_is_day = current.Variables(0).Value()
current_temperature_2m = current.Variables(1).Value()
current_precipitation = current.Variables(2).Value()
current_rain = current.Variables(3).Value()
current_cloud_cover = current.Variables(4).Value()
current_wind_speed_10m = current.Variables(5).Value()

print(f"\nCurrent time: {current.Time()}")
print(f"Current is_day: {current_is_day}")
print(f"Current temperature_2m: {current_temperature_2m}")
print(f"Current precipitation: {current_precipitation}")
print(f"Current rain: {current_rain}")
print(f"Current cloud_cover: {current_cloud_cover}")
print(f"Current wind_speed_10m: {current_wind_speed_10m}")

# Process hourly data. The order of variables needs to be the same as requested.
hourly = api_response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_direct_radiation = hourly.Variables(1).ValuesAsNumpy()
hourly_is_day = hourly.Variables(2).ValuesAsNumpy()
hourly_cloud_cover = hourly.Variables(3).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(4).ValuesAsNumpy()
hourly_shortwave_radiation = hourly.Variables(5).ValuesAsNumpy()
hourly_diffuse_radiation = hourly.Variables(6).ValuesAsNumpy()
hourly_sunshine_duration = hourly.Variables(7).ValuesAsNumpy()
hourly_cloud_cover_low = hourly.Variables(8).ValuesAsNumpy()
hourly_cloud_cover_mid = hourly.Variables(9).ValuesAsNumpy()
hourly_cloud_cover_high = hourly.Variables(10).ValuesAsNumpy()
hourly_relative_humidity_2m = hourly.Variables(11).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(12).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(13).ValuesAsNumpy()
hourly_wind_gusts_10m = hourly.Variables(14).ValuesAsNumpy()
hourly_wind_direction_10m = hourly.Variables(15).ValuesAsNumpy()
hourly_rain = hourly.Variables(16).ValuesAsNumpy()
hourly_showers = hourly.Variables(17).ValuesAsNumpy()
hourly_snowfall = hourly.Variables(18).ValuesAsNumpy()
hourly_surface_pressure = hourly.Variables(19).ValuesAsNumpy()
hourly_visibility = hourly.Variables(20).ValuesAsNumpy()
hourly_weather_code = hourly.Variables(21).ValuesAsNumpy()

hourly_data = {
    "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["direct_radiation"] = hourly_direct_radiation
hourly_data["is_day"] = hourly_is_day
hourly_data["cloud_cover"] = hourly_cloud_cover
hourly_data["precipitation"] = hourly_precipitation
hourly_data["shortwave_radiation"] = hourly_shortwave_radiation
hourly_data["diffuse_radiation"] = hourly_diffuse_radiation
hourly_data["sunshine_duration"] = hourly_sunshine_duration
hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
hourly_data["relative_humidity_2m"] = hourly_relative_humidity_2m
hourly_data["dew_point_2m"] = hourly_dew_point_2m
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m
hourly_data["wind_direction_10m"] = hourly_wind_direction_10m
hourly_data["rain"] = hourly_rain
hourly_data["showers"] = hourly_showers
hourly_data["snowfall"] = hourly_snowfall
hourly_data["surface_pressure"] = hourly_surface_pressure
hourly_data["visibility"] = hourly_visibility
hourly_data["weather_code"] = hourly_weather_code

hourly_dataframe = pd.DataFrame(data=hourly_data)
print("\nHourly data\n", hourly_dataframe)

# Process daily data. The order of variables needs to be the same as requested.
daily = api_response.Daily()
daily_daylight_duration = daily.Variables(0).ValuesAsNumpy()
daily_shortwave_radiation_sum = daily.Variables(1).ValuesAsNumpy()
daily_sunrise = daily.Variables(2).ValuesInt64AsNumpy()
daily_sunset = daily.Variables(3).ValuesInt64AsNumpy()
daily_weather_code = daily.Variables(4).ValuesAsNumpy()
daily_precipitation_sum = daily.Variables(5).ValuesAsNumpy()
daily_rain_sum = daily.Variables(6).ValuesAsNumpy()
daily_snowfall_sum = daily.Variables(7).ValuesAsNumpy()
daily_precipitation_hours = daily.Variables(8).ValuesAsNumpy()
daily_temperature_2m_max = daily.Variables(9).ValuesAsNumpy()
daily_temperature_2m_min = daily.Variables(10).ValuesAsNumpy()

daily_data = {
    "date": pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left",
    )
}

daily_data["daylight_duration"] = daily_daylight_duration
daily_data["shortwave_radiation_sum"] = daily_shortwave_radiation_sum
daily_data["sunrise"] = daily_sunrise
daily_data["sunset"] = daily_sunset
daily_data["weather_code"] = daily_weather_code
daily_data["precipitation_sum"] = daily_precipitation_sum
daily_data["rain_sum"] = daily_rain_sum
daily_data["snowfall_sum"] = daily_snowfall_sum
daily_data["precipitation_hours"] = daily_precipitation_hours
daily_data["temperature_2m_max"] = daily_temperature_2m_max
daily_data["temperature_2m_min"] = daily_temperature_2m_min

daily_dataframe = pd.DataFrame(data=daily_data)
print("\nDaily data\n", daily_dataframe)
