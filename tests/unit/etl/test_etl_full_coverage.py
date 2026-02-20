from __future__ import annotations

import types
from pathlib import Path

import pandas as pd
import pytest

from ors.etl import etl


class FakeCountryHoliday(set):
    def __init__(self, _country_code: str):
        super().__init__({pd.Timestamp("2025-01-01").date()})


def _sample_price_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Timestamp": pd.date_range("2025-01-01 00:00:00", periods=4, freq="h"),
            "Price": [10.0, 20.0, 30.0, 40.0],
            "category": ["A", "A", "B", "B"],
            "hour": [0, 1, 2, 3],
        }
    )


def _sample_weather_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Timestamp": pd.date_range("2025-01-01 00:00:00", periods=2, freq="2h"),
            "temp": [0.0, 10.0],
            "constant": [5.0, 5.0],
        }
    )


def _sample_sun_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Timestamp": pd.date_range("2025-01-01 00:00:00", periods=2, freq="2h"),
            "sunrise": ["2025-01-01 08:00:00", "2025-01-01 08:00:00"],
            "sunset": ["2025-01-01 16:00:00", "2025-01-01 16:00:00"],
            "daylight_duration": [28800, 28800],
        }
    )


def test_standardize_timestamp_column_rename_and_error() -> None:
    df = pd.DataFrame({"ts_utc": ["2025-01-01 00:00:00+00:00"]})
    out = etl.standardize_timestamp_column(df, ["timestamp", "ts_utc"])
    assert "Timestamp" in out.columns
    assert out["Timestamp"].dt.tz is None

    already = pd.DataFrame({"Timestamp": ["2025-01-01 00:00:00"]})
    out2 = etl.standardize_timestamp_column(already, ["Timestamp"])
    assert "Timestamp" in out2.columns

    with pytest.raises(ValueError, match="Missing timestamp column"):
        etl.standardize_timestamp_column(pd.DataFrame({"x": [1]}), ["ts", "date"])


def test_generate_time_data_and_transform_time_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        __import__("sys").modules,
        "holidays",
        types.SimpleNamespace(CountryHoliday=FakeCountryHoliday),
    )

    time_df = etl.generate_time_data("2025-01-01 00:00:00", "2025-01-01 03:00:00", "UK")
    assert {"IsWeekend", "IsHoliday", "IsWorkingDay"}.issubset(time_df.columns)

    transformed = etl.transform_time_data(time_df.copy())
    assert "Timestamp" in transformed.columns
    assert "Hour_sin" in transformed.columns
    assert "Hour" not in transformed.columns


def test_transform_weather_sun_price_and_helpers(capsys: pytest.CaptureFixture[str]) -> None:
    weather = etl.transform_weather_data(_sample_weather_df())
    assert len(weather) == 3
    assert weather["temp"].between(0.0, 1.0).all()
    assert (weather["constant"] == 0.0).all()

    sun = etl.transform_sun_data(_sample_sun_df())
    assert "Solar_intensity" in sun.columns
    assert "sunrise" not in sun.columns

    price = _sample_price_df()
    transformed_price = etl.transform_price_data(price)
    assert "hour" not in transformed_price.columns
    assert len(transformed_price) == 4

    transformed_price_no_drop = etl.transform_price_data(
        pd.DataFrame(
            {
                "Timestamp": pd.date_range("2025-01-01", periods=2, freq="h"),
                "Price": [1.0, 2.0],
            }
        )
    )
    assert "Price" in transformed_price_no_drop.columns

    start_date, end_date = etl.get_timestamp_range(transformed_price)
    assert start_date <= end_date

    filtered = etl.filter_by_timestamp_range(transformed_price, start_date, end_date)
    assert len(filtered) == len(transformed_price)

    etl.log_dataset_ranges({"price_data": (start_date, end_date)})
    captured = capsys.readouterr().out
    assert "Timestamp ranges used for merge" in captured

    merged = etl.merge_datasets(
        transformed_price,
        weather,
        sun,
        pd.DataFrame(
            {"Timestamp": transformed_price["Timestamp"], "flag": [True] * len(transformed_price)}
        ),
    )
    assert "Solar_intensity" in merged.columns


def test_transform_weather_data_numeric_timestamp_hits_continue_branch() -> None:
    weather_data = pd.DataFrame(
        {
            "Timestamp": [1, 2],
            "temp": [10.0, 20.0],
        }
    )
    with pytest.raises(TypeError):
        etl.transform_weather_data(weather_data)


def test_add_lagged_features_success_and_error() -> None:
    base = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2025-01-01", periods=5, freq="h"),
            "Price": [1, 2, 3, 4, 5],
            "IsWeekend": [False, False, False, False, False],
        }
    )
    out = etl.add_lagged_features(base, lag_steps=(1, 2), drop_na=True)
    assert "Price_lag_1h" in out.columns
    assert len(out) == 3

    out_no_drop = etl.add_lagged_features(base, lag_steps=(), drop_na=True)
    assert len(out_no_drop) == len(base)

    with pytest.raises(ValueError, match="Timestamp column missing"):
        etl.add_lagged_features(pd.DataFrame({"x": [1]}), lag_steps=(1,))


def test_preprocess_merge_success_and_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        __import__("sys").modules,
        "holidays",
        types.SimpleNamespace(CountryHoliday=FakeCountryHoliday),
    )

    price = _sample_price_df()
    weather = _sample_weather_df()
    sun = _sample_sun_df()

    merged = etl.preprocess_merge(
        price.copy(), weather.copy(), sun.copy(), lag_steps=(1,), drop_na=False
    )
    assert "Price" in merged.columns

    bad_price = price.copy()
    bad_price["Timestamp"] = pd.NaT
    with pytest.raises(ValueError, match="Invalid timestamp range"):
        etl.preprocess_merge(bad_price, weather.copy(), sun.copy())

    shifted_sun = sun.copy()
    shifted_sun["Timestamp"] = pd.date_range("2030-01-01", periods=2, freq="2h")
    with pytest.raises(ValueError, match="No overlapping timestamp range"):
        etl.preprocess_merge(price.copy(), weather.copy(), shifted_sun)


def test_main_writes_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_dir = tmp_path / "src" / "ors" / "etl"
    data_dir = tmp_path / "src" / "ors" / "Data"
    module_dir.mkdir(parents=True)
    data_dir.mkdir(parents=True)

    (data_dir / "price_data.csv").write_text(
        "timestamp,Price\n2025-01-01 00:00:00,10\n2025-01-01 01:00:00,20\n"
    )
    (data_dir / "historical_hourly_2025.csv").write_text(
        "timestamp_utc,temp,constant\n2025-01-01 00:00:00,0,5\n2025-01-01 02:00:00,10,5\n"
    )
    (data_dir / "historical_daily_2025.csv").write_text(
        "date_utc,sunrise,sunset,daylight_duration\n"
        "2025-01-01 00:00:00,2025-01-01 08:00:00,2025-01-01 16:00:00,28800\n"
        "2025-01-01 02:00:00,2025-01-01 08:00:00,2025-01-01 16:00:00,28800\n"
    )

    monkeypatch.setitem(
        __import__("sys").modules,
        "holidays",
        types.SimpleNamespace(CountryHoliday=FakeCountryHoliday),
    )
    monkeypatch.setattr(etl.Path, "resolve", lambda _self: module_dir / "etl.py")

    monkeypatch.chdir(tmp_path)
    etl.main()

    assert (tmp_path / "expected_columns.txt").exists()
    assert (data_dir / "merged_dataset.csv").exists()
