from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from ors.clients.weather_client import (
    DEFAULT_PARAMS,
    WeatherFetcherError,
    make_client,
    solar_radiance_15_mins,
    to_hourly_df,
)
from ors.config.api_endpoints import ARCHIVE_API_URL, FORECAST_API_URL


class FakeHourlyVar:
    def __init__(self, arr):
        self._arr = np.array(arr)

    def values_as_numpy(self):
        return self._arr

    def __getattr__(self, name):
        # Support Open-Meteo-style method name
        if name == "ValuesAsNumpy":
            return self.values_as_numpy
        raise AttributeError(name)


class FakeHourlyBlock:
    def __init__(self, start, interval, arrays):
        self._start = start
        self._interval = interval
        self._arrays = arrays
        self._n = len(arrays[0]) if arrays else 0

    def time(self):
        return self._start

    def time_end(self):
        return self._start + self._interval * self._n

    def interval(self):
        return self._interval

    def variables(self, i):
        return FakeHourlyVar(self._arrays[i])

    def __getattr__(self, name):
        mapping = {
            "Time": self.time,
            "TimeEnd": self.time_end,
            "Interval": self.interval,
            "Variables": self.variables,
        }
        if name in mapping:
            return mapping[name]
        raise AttributeError(name)


class FakeResponse:
    def __init__(self, hourly_block):
        self._hourly = hourly_block

    def hourly(self):
        return self._hourly

    def __getattr__(self, name):
        # Support Open-Meteo-style response method
        if name == "Hourly":
            return self.hourly
        raise AttributeError(name)


def test_to_hourly_df_ok():
    start = 0
    interval = 3600

    arrays = [
        [1.0, 2.0, 3.0],  # temperature_2m
        [0.0, 0.1, 0.2],  # precipitation
    ]

    hourly_block = FakeHourlyBlock(start, interval, arrays)
    response = FakeResponse(hourly_block)

    df = to_hourly_df(response, ["temperature_2m", "precipitation"])

    assert list(df.columns) == ["timestamp_utc", "temperature_2m", "precipitation"]
    assert len(df) == 3
    assert pd.api.types.is_datetime64tz_dtype(df["timestamp_utc"])
    assert df["temperature_2m"].tolist() == [1.0, 2.0, 3.0]
    assert df["precipitation"].tolist() == [0.0, 0.1, 0.2]


def test_to_hourly_df_missing_variable():
    start = 0
    interval = 3600

    arrays = [
        [1.0, 2.0, 3.0],  # only ONE variable provided
    ]

    hourly_block = FakeHourlyBlock(start, interval, arrays)
    response = FakeResponse(hourly_block)

    with pytest.raises((WeatherFetcherError, IndexError)):
        to_hourly_df(response, ["temperature_2m", "precipitation"])


def test_to_hourly_df_no_hourly_block():
    class EmptyResponse:
        pass

    with pytest.raises(WeatherFetcherError, match="no hourly block"):
        to_hourly_df(EmptyResponse(), ["temperature_2m"])


def test_make_client_returns_client():
    client = make_client()
    assert client is not None
    assert hasattr(client, "weather_api")


def test_api_endpoints_are_valid_urls():
    assert FORECAST_API_URL.startswith("https://")
    assert ARCHIVE_API_URL.startswith("https://")
    assert "open-meteo" in FORECAST_API_URL
    assert "open-meteo" in ARCHIVE_API_URL


def test_fetch_forecast_raises_on_empty_response(monkeypatch):
    from ors.clients import weather_client

    class FakeClient:
        def weather_api(self, url, params=None):
            return []

    with pytest.raises(WeatherFetcherError, match="No responses"):
        weather_client.fetch_forecast(FakeClient(), {"hourly": []})


# DEFAULT_PARAMS tests
class TestDefaultParams:
    def test_latitude(self):
        assert DEFAULT_PARAMS["latitude"] == pytest.approx(54.727592)

    def test_longitude(self):
        assert DEFAULT_PARAMS["longitude"] == pytest.approx(-2.6679993)

    def test_model(self):
        assert DEFAULT_PARAMS["models"] == "ukmo_uk_deterministic_2km"

    def test_timezone(self):
        assert DEFAULT_PARAMS["timezone"] == "GMT"

    def test_forecast_days(self):
        assert DEFAULT_PARAMS["forecast_days"] == 1

    def test_forecast_hours(self):
        assert DEFAULT_PARAMS["forecast_hours"] == 24

    def test_past_hours(self):
        assert DEFAULT_PARAMS["past_hours"] == 6

    def test_hourly_vars_exact(self):
        assert DEFAULT_PARAMS["hourly"] == [
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
        ]

    def test_hourly_vars_count(self):
        assert len(DEFAULT_PARAMS["hourly"]) == 18

    def test_daily_vars_exact(self):
        assert DEFAULT_PARAMS["daily"] == [
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
        ]

    def test_daily_vars_count(self):
        assert len(DEFAULT_PARAMS["daily"]) == 11

    def test_current_vars_exact(self):
        assert DEFAULT_PARAMS["current"] == [
            "is_day",
            "temperature_2m",
            "precipitation",
            "rain",
            "cloud_cover",
            "cloud_cover_low",
            "wind_speed_10m",
        ]

    def test_current_vars_count(self):
        assert len(DEFAULT_PARAMS["current"]) == 7

    def test_has_all_expected_keys(self):
        expected_keys = {
            "latitude",
            "longitude",
            "hourly",
            "daily",
            "current",
            "models",
            "timezone",
            "forecast_days",
            "forecast_hours",
            "past_hours",
        }
        assert set(DEFAULT_PARAMS.keys()) == expected_keys

    def test_no_duplicate_hourly_vars(self):
        assert len(DEFAULT_PARAMS["hourly"]) == len(set(DEFAULT_PARAMS["hourly"]))

    def test_no_duplicate_daily_vars(self):
        assert len(DEFAULT_PARAMS["daily"]) == len(set(DEFAULT_PARAMS["daily"]))

    def test_no_duplicate_current_vars(self):
        assert len(DEFAULT_PARAMS["current"]) == len(set(DEFAULT_PARAMS["current"]))


# solar_radiance_15_mins tests
#
# Shared helpers for building fake Open-Meteo minutely_15 responses.
# The API block starts at `block_start_epoch` and contains `values` at 15-min
# intervals.  `values` may include float("nan") to simulate null predictions.


def _make_fake_client(block_start_epoch, values, captured=None):
    """Return a FakeClient whose weather_api returns a single minutely/hourly response."""

    class FakeVar:
        def values_as_numpy(self):
            return np.array(values, dtype=np.float64)

        def __getattr__(self, name):
            if name == "ValuesAsNumpy":
                return self.values_as_numpy
            raise AttributeError(name)

    class FakeBlock:
        def time(self):
            return block_start_epoch

        def time_end(self):
            return block_start_epoch + len(values) * 900

        def interval(self):
            return 900

        def variables(self, i):
            return FakeVar()

        def __getattr__(self, name):
            mapping = {
                "Time": self.time,
                "TimeEnd": self.time_end,
                "Interval": self.interval,
                "Variables": self.variables,
            }
            if name in mapping:
                return mapping[name]
            raise AttributeError(name)

    class FakeResponse:
        def minutely_15(self):
            return FakeBlock()

        def hourly(self):
            return FakeBlock()

        def __getattr__(self, name):
            if name == "Minutely15":
                return self.minutely_15
            if name == "Hourly":
                return self.hourly
            raise AttributeError(name)

    class FakeClient:
        def weather_api(self, url, params=None):
            if captured is not None:
                captured["params"] = params
                captured["url"] = url
            return [FakeResponse()]

    return FakeClient()


# Midnight 2026-03-01 00:00 UTC as epoch seconds
_MIDNIGHT = int(datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc).timestamp())


class TestSolarRadiance15Mins:
    # ---- API params tests ------------------------------------------------

    def test_params_latitude(self):
        captured = {}
        start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 45, tzinfo=timezone.utc)
        client = _make_fake_client(_MIDNIGHT, [100.0] * 4, captured)
        solar_radiance_15_mins(client, start, end)
        assert captured["params"]["latitude"] == pytest.approx(54.727592)

    def test_params_longitude(self):
        captured = {}
        start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 45, tzinfo=timezone.utc)
        client = _make_fake_client(_MIDNIGHT, [100.0] * 4, captured)
        solar_radiance_15_mins(client, start, end)
        assert captured["params"]["longitude"] == pytest.approx(-2.6679993)

    def test_params_hourly_var(self):
        # Historical dates use the archive path; radiation is requested as hourly.
        captured = {}
        start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 45, tzinfo=timezone.utc)
        client = _make_fake_client(_MIDNIGHT, [100.0] * 4, captured)
        solar_radiance_15_mins(client, start, end)
        assert captured["params"]["hourly"] == ["shortwave_radiation"]

    def test_params_timezone(self):
        captured = {}
        start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 45, tzinfo=timezone.utc)
        client = _make_fake_client(_MIDNIGHT, [100.0] * 4, captured)
        solar_radiance_15_mins(client, start, end)
        assert captured["params"]["timezone"] == "GMT"

    def test_params_has_all_expected_keys(self):
        # Historical dates use the archive API; check archive request keys.
        captured = {}
        start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 45, tzinfo=timezone.utc)
        client = _make_fake_client(_MIDNIGHT, [100.0] * 4, captured)
        solar_radiance_15_mins(client, start, end)
        expected_keys = {
            "latitude",
            "longitude",
            "start_date",
            "end_date",
            "hourly",
            "timezone",
        }
        assert set(captured["params"].keys()) == expected_keys

    def test_uses_archive_api_url(self):
        # Historical dates should use the archive API, not the forecast API.
        captured = {}
        start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 45, tzinfo=timezone.utc)
        client = _make_fake_client(_MIDNIGHT, [100.0] * 4, captured)
        solar_radiance_15_mins(client, start, end)
        assert captured["url"] == ARCHIVE_API_URL

    def test_archive_uses_date_range_params(self):
        # Archive path sends start_date and end_date, not timestep counts.
        captured = {}
        start = datetime(2026, 3, 1, 16, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 18, 0, tzinfo=timezone.utc)
        n_slots = 73
        client = _make_fake_client(_MIDNIGHT, [100.0] * n_slots, captured)
        solar_radiance_15_mins(client, start, end)
        assert captured["params"]["start_date"] == "2026-03-01"
        assert captured["params"]["end_date"] == "2026-03-01"

    # ---- Output shape and content tests ----------------------------------

    def test_returns_dataframe_with_correct_columns(self):
        start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 45, tzinfo=timezone.utc)
        client = _make_fake_client(_MIDNIGHT, [100.0, 200.0, 300.0, 400.0])
        df = solar_radiance_15_mins(client, start, end)
        assert list(df.columns) == ["timestamp_utc", "shortwave_radiation"]

    def test_timestamps_are_utc_aware(self):
        start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 45, tzinfo=timezone.utc)
        client = _make_fake_client(_MIDNIGHT, [100.0, 200.0, 300.0, 400.0])
        df = solar_radiance_15_mins(client, start, end)
        assert df["timestamp_utc"].dt.tz is not None

    def test_radiation_values_match_input(self):
        start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 45, tzinfo=timezone.utc)
        expected = [100.0, 200.0, 300.0, 400.0]
        client = _make_fake_client(_MIDNIGHT, expected)
        df = solar_radiance_15_mins(client, start, end)
        assert df["shortwave_radiation"].tolist() == pytest.approx(expected)

    def test_timestamp_interval_is_15_minutes(self):
        start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 45, tzinfo=timezone.utc)
        client = _make_fake_client(_MIDNIGHT, [100.0, 200.0, 300.0, 400.0])
        df = solar_radiance_15_mins(client, start, end)
        diffs = df["timestamp_utc"].diff().dropna()
        assert all(d == pd.Timedelta(minutes=15) for d in diffs)

    # ---- Trimming tests --------------------------------------------------

    def test_trims_rows_before_start_datetime(self):
        # API returns from midnight, but start is 00:30 → first two rows dropped
        start = datetime(2026, 3, 1, 0, 30, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 45, tzinfo=timezone.utc)
        #           00:00  00:15  00:30  00:45
        values = [10.0, 20.0, 30.0, 40.0]
        client = _make_fake_client(_MIDNIGHT, values)
        df = solar_radiance_15_mins(client, start, end)
        assert len(df) == 2
        assert df["shortwave_radiation"].tolist() == pytest.approx([30.0, 40.0])

    def test_trims_rows_after_end_datetime(self):
        # API returns 4 slots but end is 00:15 → last two rows dropped
        start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 15, tzinfo=timezone.utc)
        values = [10.0, 20.0, 30.0, 40.0]
        client = _make_fake_client(_MIDNIGHT, values)
        df = solar_radiance_15_mins(client, start, end)
        assert len(df) == 2
        assert df["shortwave_radiation"].tolist() == pytest.approx([10.0, 20.0])

    def test_trims_both_sides(self):
        # Start at 00:15, end at 00:30 → only middle two rows kept
        start = datetime(2026, 3, 1, 0, 15, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 30, tzinfo=timezone.utc)
        values = [10.0, 20.0, 30.0, 40.0]
        client = _make_fake_client(_MIDNIGHT, values)
        df = solar_radiance_15_mins(client, start, end)
        assert len(df) == 2
        assert df["shortwave_radiation"].tolist() == pytest.approx([20.0, 30.0])

    def test_start_at_4pm_trims_early_hours(self):
        # Simulate: API returns from midnight, start at 16:00
        n_slots = 68  # midnight to 16:45 = 68 slots
        values = list(range(n_slots))
        start = datetime(2026, 3, 1, 16, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 16, 45, tzinfo=timezone.utc)
        client = _make_fake_client(_MIDNIGHT, values)
        df = solar_radiance_15_mins(client, start, end)
        # 16:00 is slot index 64 (16*4), 16:45 is index 67
        assert len(df) == 4
        assert df["shortwave_radiation"].tolist() == pytest.approx([64.0, 65.0, 66.0, 67.0])

    # ---- Null handling tests ---------------------------------------------

    def test_raises_when_start_datetime_value_is_null(self):
        # First value at start_datetime is NaN → too far in the future
        start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 45, tzinfo=timezone.utc)
        values = [float("nan"), 200.0, 300.0, 400.0]
        client = _make_fake_client(_MIDNIGHT, values)
        with pytest.raises(WeatherFetcherError, match="too far in the future"):
            solar_radiance_15_mins(client, start, end)

    def test_drops_intermediate_null_rows(self):
        # Valid first row, some nulls in the middle; archive path interpolates between
        # surrounding values, so the null slot is filled rather than dropped.
        start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 45, tzinfo=timezone.utc)
        values = [100.0, float("nan"), 300.0, 400.0]
        client = _make_fake_client(_MIDNIGHT, values)
        df = solar_radiance_15_mins(client, start, end)
        assert len(df) == 4
        assert df["shortwave_radiation"].tolist() == pytest.approx([100.0, 200.0, 300.0, 400.0])

    def test_drops_trailing_null_rows(self):
        start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 45, tzinfo=timezone.utc)
        values = [100.0, 200.0, float("nan"), float("nan")]
        client = _make_fake_client(_MIDNIGHT, values)
        df = solar_radiance_15_mins(client, start, end)
        assert len(df) == 2
        assert df["shortwave_radiation"].tolist() == pytest.approx([100.0, 200.0])

    def test_raises_when_all_values_in_window_are_null(self):
        start = datetime(2026, 3, 1, 0, 30, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 45, tzinfo=timezone.utc)
        # First two are before start (trimmed), remaining two are NaN
        values = [100.0, 200.0, float("nan"), float("nan")]
        client = _make_fake_client(_MIDNIGHT, values)
        # After trimming, only NaN rows remain; first in-window row is NaN
        with pytest.raises(WeatherFetcherError, match="too far in the future"):
            solar_radiance_15_mins(client, start, end)

    # ---- Error handling tests --------------------------------------------

    def test_raises_on_empty_api_response(self):
        class EmptyClient:
            def weather_api(self, url, params=None):
                return []

        start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 45, tzinfo=timezone.utc)
        with pytest.raises(WeatherFetcherError, match="No historical hourly"):
            solar_radiance_15_mins(EmptyClient(), start, end)

    def test_raises_on_api_exception_with_context(self):
        class ExplodingClient:
            def weather_api(self, url, params=None):
                raise RuntimeError("Forecast API error: too many timesteps requested")

        start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        end = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
        with pytest.raises(WeatherFetcherError, match="Open-Meteo API request failed") as exc_info:
            solar_radiance_15_mins(ExplodingClient(), start, end)
        # The message should include the start, end, and timesteps for traceability
        msg = str(exc_info.value)
        assert "start=" in msg
        assert "end=" in msg
        assert "timesteps=" in msg
        # The original exception should be chained
        assert exc_info.value.__cause__ is not None
        assert "too many timesteps" in str(exc_info.value.__cause__)

    def test_raises_when_start_after_end(self):
        start = datetime(2026, 3, 1, 1, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        client = _make_fake_client(_MIDNIGHT, [100.0] * 8)
        with pytest.raises(WeatherFetcherError, match="must not be after"):
            solar_radiance_15_mins(client, start, end)

    def test_index_is_reset_after_filtering(self):
        # After trimming and dropping nulls, index should be 0-based
        start = datetime(2026, 3, 1, 0, 15, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 0, 45, tzinfo=timezone.utc)
        values = [10.0, 20.0, float("nan"), 40.0]
        client = _make_fake_client(_MIDNIGHT, values)
        df = solar_radiance_15_mins(client, start, end)
        assert list(df.index) == list(range(len(df)))

    # ---- Default datetime tests ------------------------------------------

    def test_defaults_start_to_now_utc(self, monkeypatch):
        frozen_now = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(
            "ors.clients.weather_client.datetime",
            type(
                "FakeDatetime",
                (datetime,),
                {
                    "now": classmethod(lambda cls, tz=None: frozen_now),
                },
            ),
        )

        # API block starts at midnight, need slots up to 10:00+48h but we
        # only care that the captured params reflect the frozen time.
        captured = {}
        # 58h from midnight = 232 slots + 1 = 233 timesteps needed
        n_slots = 233
        epoch_midnight = int(datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc).timestamp())
        client = _make_fake_client(epoch_midnight, [100.0] * n_slots, captured)

        df = solar_radiance_15_mins(client)

        # Should have trimmed rows before 10:00
        first_ts = df.iloc[0]["timestamp_utc"]
        assert first_ts == pd.Timestamp("2026-03-01 10:00:00", tz="UTC")

    def test_defaults_end_to_48h_after_start(self, monkeypatch):
        frozen_now = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
        monkeypatch.setattr(
            "ors.clients.weather_client.datetime",
            type(
                "FakeDatetime",
                (datetime,),
                {
                    "now": classmethod(lambda cls, tz=None: frozen_now),
                },
            ),
        )

        captured = {}
        n_slots = 233
        epoch_midnight = int(datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc).timestamp())
        client = _make_fake_client(epoch_midnight, [100.0] * n_slots, captured)

        df = solar_radiance_15_mins(client)

        # Last row should be at or before 10:00 + 48h = March 3 10:00
        last_ts = df.iloc[-1]["timestamp_utc"]
        assert last_ts <= pd.Timestamp("2026-03-03 10:00:00", tz="UTC")

    def test_explicit_start_defaults_end_to_48h_later(self):
        start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
        # end should default to start + 48h = March 3 00:00
        # Need slots from midnight to March 3 00:00 = 192 + 1 = 193
        n_slots = 193
        client = _make_fake_client(_MIDNIGHT, [100.0] * n_slots)

        df = solar_radiance_15_mins(client, start_datetime=start)

        last_ts = df.iloc[-1]["timestamp_utc"]
        assert last_ts <= pd.Timestamp("2026-03-03 00:00:00", tz="UTC")
        assert len(df) == 193
