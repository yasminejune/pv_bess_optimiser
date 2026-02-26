"""Fetch forecast and historical weather data from Open-Meteo and save to CSV.

Usage:
    python scripts/fetch_weather_data.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ors.clients.weather_client import (
    DEFAULT_PARAMS,
    fetch_forecast,
    fetch_hist_daily,
    fetch_hist_hourly,
    make_client,
    solar_radiance_15_mins,
    to_current,
    to_daily_df,
    to_hourly_df,
)

# Ensure the project root (parent of scripts/) is on sys.path so that
# `ors` is importable regardless of how or where this script is invoked.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))


def main() -> None:
    """Run the full forecast and historical data pipeline and save the datasets to CSV."""
    data_dir = _PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

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

    # 15-minute solar radiance forecast (next 48 hours from now)
    now = datetime.now(timezone.utc)
    solar_15m_df = solar_radiance_15_mins(
        client, now + timedelta(hours=40), now + timedelta(days=10)
    )
    print(f"\nSolar radiance 15-min rows: {len(solar_15m_df)}")

    # Save files
    hourly_path = data_dir / "historical_hourly_2023_2025.csv"
    daily_path = data_dir / "historical_daily_2023_2025.csv"
    solar_15m_path = data_dir / "solar_radiance_15min_forecast.csv"

    hist_hourly.to_csv(hourly_path, index=False)
    hist_daily.to_csv(daily_path, index=False)
    solar_15m_df.to_csv(solar_15m_path, index=False)

    print(f"\nSaved: {hourly_path}")
    print(f"Saved: {daily_path}")
    print(f"Saved: {solar_15m_path}")
    print("\nHistorical hourly rows:", len(hist_hourly))
    print("Historical daily rows:", len(hist_daily))
    print("Solar radiance 15-min rows:", len(solar_15m_df))


if __name__ == "__main__":
    main()
