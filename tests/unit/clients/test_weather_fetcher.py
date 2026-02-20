import numpy as np
import pandas as pd
import pytest

from ors.clients.weather_fetcher import WeatherFetcherError, build_hourly_dataframe


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
