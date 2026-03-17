import numpy as np
import pandas as pd
import pytest

from ors.clients.weather_fetcher import (
    WeatherFetcherError,
    build_hourly_dataframe,
    fetch_forecast,
    fetch_hist_daily,
    fetch_hist_hourly,
    to_current,
    to_daily_df,
    to_hourly_df,
)


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


def test_build_hourly_dataframe_ok():
    start = 0
    interval = 3600

    arrays = [
        [1.0, 2.0, 3.0],  # temperature_2m
        [0.0, 0.1, 0.2],  # precipitation
    ]

    hourly_block = FakeHourlyBlock(start, interval, arrays)
    response = FakeResponse(hourly_block)

    df = build_hourly_dataframe(response, ["temperature_2m", "precipitation"])

    assert list(df.columns) == ["timestamp_utc", "temperature_2m", "precipitation"]
    assert len(df) == 3
    assert pd.api.types.is_datetime64tz_dtype(df["timestamp_utc"])
    assert df["temperature_2m"].tolist() == [1.0, 2.0, 3.0]
    assert df["precipitation"].tolist() == [0.0, 0.1, 0.2]


def test_build_hourly_dataframe_missing():
    start = 0
    interval = 3600

    arrays = [
        [1.0, 2.0, 3.0],  # only ONE variable provided
    ]

    hourly_block = FakeHourlyBlock(start, interval, arrays)
    response = FakeResponse(hourly_block)

    with pytest.raises((WeatherFetcherError, IndexError)):
        build_hourly_dataframe(response, ["temperature_2m", "precipitation"])


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class FakeDailyVar:
    def __init__(self, arr):
        self._arr = np.array(arr, dtype=np.float64)

    def ValuesAsNumpy(self):
        return self._arr

    def ValuesInt64AsNumpy(self):
        return self._arr.astype(np.int64)

    def Value(self):
        return float(self._arr[0])


class FakeDailyBlock:
    def __init__(self, start, end, interval, variables):
        self._start = start
        self._end = end
        self._interval = interval
        self._variables = variables

    def Time(self):
        return self._start

    def TimeEnd(self):
        return self._end

    def Interval(self):
        return self._interval

    def Variables(self, i):
        return self._variables[i]


class FakeCurrentBlock:
    def __init__(self, time_unix, variables):
        self._time_unix = time_unix
        self._variables = variables

    def Time(self):
        return self._time_unix

    def Variables(self, i):
        return self._variables[i]


class FakeFullResponse:
    """Response that supports Hourly(), Daily(), and Current()."""

    def __init__(self, hourly=None, daily=None, current=None):
        self._hourly = hourly
        self._daily = daily
        self._current = current

    def Hourly(self):
        return self._hourly

    def Daily(self):
        return self._daily

    def Current(self):
        return self._current


def _hourly_block(n_hours=24, var_count=1):
    start = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())
    end = start + n_hours * 3600
    variables = [FakeDailyVar(np.ones(n_hours) * (i + 10)) for i in range(var_count)]
    return FakeHourlyBlock(start, 3600, [np.ones(n_hours) * (i + 10) for i in range(var_count)])


def _daily_block(n_days=7, var_count=1):
    start = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())
    end = start + n_days * 86400
    variables = [FakeDailyVar(np.ones(n_days) * (i + 20)) for i in range(var_count)]
    return FakeDailyBlock(start, end, 86400, variables)


class TestToDailyDf:
    def test_basic_daily(self):
        block = _daily_block(7, 1)
        resp = FakeFullResponse(daily=block)
        df = to_daily_df(resp, ["temp_max"])
        assert len(df) == 7
        assert "date_utc" in df.columns
        assert "temp_max" in df.columns

    def test_sunrise_sunset(self):
        n_days = 3
        start = int(pd.Timestamp("2026-01-01", tz="UTC").timestamp())
        end = start + n_days * 86400
        sunrise_vals = np.array([start + i * 86400 + 7 * 3600 for i in range(n_days)], dtype=float)
        variables = [FakeDailyVar(sunrise_vals)]
        block = FakeDailyBlock(start, end, 86400, variables)
        resp = FakeFullResponse(daily=block)
        df = to_daily_df(resp, ["sunrise"])
        assert pd.api.types.is_datetime64_any_dtype(df["sunrise"])


class TestToCurrent:
    def test_basic_current(self):
        time_unix = int(pd.Timestamp("2026-01-01 12:00", tz="UTC").timestamp())
        variables = [FakeDailyVar([22.5]), FakeDailyVar([0.0])]
        block = FakeCurrentBlock(time_unix, variables)
        resp = FakeFullResponse(current=block)
        result = to_current(resp, ["temperature_2m", "precipitation"])
        assert result["time_unix"] == time_unix
        assert "time_utc" in result
        assert result["temperature_2m"] == pytest.approx(22.5)


class TestFetchForecast:
    def test_returns_first_response(self):
        resp_obj = FakeFullResponse()

        class FakeClient:
            def weather_api(self, url, params):
                return [resp_obj]

        result = fetch_forecast(FakeClient())
        assert result is resp_obj

    def test_empty_response_raises(self):
        class FakeClient:
            def weather_api(self, url, params):
                return []

        with pytest.raises(WeatherFetcherError, match="No responses"):
            fetch_forecast(FakeClient())

    def test_uses_default_params(self):
        captured = {}

        class FakeClient:
            def weather_api(self, url, params):
                captured["params"] = params
                return [FakeFullResponse()]

        fetch_forecast(FakeClient())
        assert "latitude" in captured["params"]


class TestFetchHistHourly:
    def test_returns_dataframe(self):
        block = _hourly_block(48, 1)
        resp = FakeResponse(block)

        class FakeClient:
            def weather_api(self, url, params):
                return [resp]

        df = fetch_hist_hourly(FakeClient(), 54.7, -2.6, "2026-01-01", "2026-01-02", ["temp"])
        assert isinstance(df, pd.DataFrame)

    def test_empty_response_raises(self):
        class FakeClient:
            def weather_api(self, url, params):
                return []

        with pytest.raises(WeatherFetcherError, match="No historical hourly"):
            fetch_hist_hourly(FakeClient(), 54.7, -2.6, "2026-01-01", "2026-01-02", ["temp"])


class TestFetchHistDaily:
    def test_empty_response_raises(self):
        class FakeClient:
            def weather_api(self, url, params):
                return []

        with pytest.raises(WeatherFetcherError, match="No historical daily"):
            fetch_hist_daily(FakeClient(), 54.7, -2.6, "2026-01-01", "2026-01-07", ["temp_max"])


class TestNoHourlyBlock:
    def test_raises_when_no_hourly_attribute(self):
        with pytest.raises(WeatherFetcherError, match="no hourly block"):
            to_hourly_df(object(), ["temp"])


class TestLengthMismatch:
    def test_raises_on_mismatched_values(self):
        start = 0
        interval = 3600
        arrays = [[1.0, 2.0]]  # 2 values
        block = FakeHourlyBlock(start, interval, arrays)
        block._n = 5  # 5 timestamps
        resp = FakeResponse(block)
        with pytest.raises(WeatherFetcherError, match="Length mismatch"):
            to_hourly_df(resp, ["temp"])
