import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ors.services.prediction.data_pipeline import (
    get_feature_cols,
    load_source_data,
    preprocess_raw_data,
)
from ors.services.prediction.prediction_model import (
    predict_prices,
    prepare_features,
    time_based_split,
)


class DummyModel:
    def __init__(self, value: float = 0.0):
        self.value = value

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.full(len(features), self.value, dtype=float)


class PredictionModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            {
                "Timestamp": pd.date_range("2025-01-01", periods=5, freq="h"),
                "Price": [10.0, 11.0, 12.0, 13.0, 14.0],
                "MaxTemp": [1.0, 2.0, 3.0, 4.0, 5.0],
                "IsWeekend": [False, False, False, False, False],
                "Humidity": [0.5, 0.6, 0.7, 0.8, 0.9],
            }
        )

    def test_prepare_features_drops_timestamp_and_target(self):
        features, target = prepare_features(self.df, target_col="Price")
        self.assertNotIn("Timestamp", features.columns)
        self.assertNotIn("Price", features.columns)
        self.assertEqual(len(target), len(self.df))

    def test_prepare_features_casts_bool(self):
        features, _ = prepare_features(self.df, target_col="Price")
        self.assertTrue(np.issubdtype(features["IsWeekend"].dtype, np.integer))

    def test_time_based_split_invalid(self):
        features, target = prepare_features(self.df, target_col="Price")
        with self.assertRaises(ValueError):
            time_based_split(features, target, test_size=1.0)

    def test_predict_prices_uses_inference_prep(self):
        model = DummyModel(value=42.0)
        preds = predict_prices(model, self.df, target_col="Price")
        self.assertEqual(len(preds), len(self.df))
        self.assertTrue(np.all(preds == 42.0))


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------
# Helpers for data pipeline tests
# ---------------------------------------------------------------------------


def _price_df(n: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
            "price": np.linspace(50.0, 100.0, n),
        }
    )


def _hourly_df(n: int = 8) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-01-01", periods=n, freq="h", tz="UTC"),
            "temperature_2m": np.ones(n) * 10.0,
            "relative_humidity_2m": np.ones(n) * 60.0,
            "dew_point_2m": np.ones(n) * 5.0,
            "precipitation": np.zeros(n),
            "rain": np.zeros(n),
            "snowfall": np.zeros(n),
            "cloud_cover": np.zeros(n),
            "cloud_cover_low": np.zeros(n),
            "cloud_cover_mid": np.zeros(n),
            "cloud_cover_high": np.zeros(n),
            "shortwave_radiation": np.ones(n),
            "direct_radiation": np.ones(n),
            "wind_speed_10m": np.ones(n),
            "wind_gusts_10m": np.ones(n),
            "wind_direction_10m": np.ones(n) * 180.0,
            "surface_pressure": np.ones(n) * 1013.0,
            "weather_code": np.zeros(n),
        }
    )


def _daily_df(n: int = 2) -> pd.DataFrame:
    base = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "date_utc": base,
            "weather_code": np.zeros(n),
            "sunrise": (base + pd.Timedelta("6h")).astype(str),
            "sunset": (base + pd.Timedelta("18h")).astype(str),
            "daylight_duration": np.full(n, 43200.0),
            "shortwave_radiation_sum": np.ones(n),
            "precipitation_sum": np.zeros(n),
            "rain_sum": np.zeros(n),
            "snowfall_sum": np.zeros(n),
            "precipitation_hours": np.zeros(n),
            "temperature_2m_max": np.ones(n) * 15.0,
            "temperature_2m_min": np.ones(n) * 5.0,
        }
    )


# ---------------------------------------------------------------------------
# get_feature_cols
# ---------------------------------------------------------------------------


def test_get_feature_cols_excludes_bookkeeping_columns():
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2025-01-01", periods=3, freq="h"),
            "price": [1.0, 2.0, 3.0],
            "price_raw": [1.0, 2.0, 3.0],
            "is_price_patched": [0, 0, 0],
            "time_idx": [0, 1, 2],
            "temperature_2m": [10.0, 11.0, 12.0],
            "hour": [0, 1, 2],
        }
    )

    cols = get_feature_cols(df)

    assert "Timestamp" not in cols
    assert "price" not in cols
    assert "price_raw" not in cols
    assert "is_price_patched" not in cols
    assert "time_idx" not in cols
    assert "temperature_2m" in cols
    assert "hour" in cols


def test_get_feature_cols_returns_list():
    df = pd.DataFrame({"Timestamp": [], "price": [], "feat_a": [], "feat_b": []})
    cols = get_feature_cols(df)
    assert isinstance(cols, list)


# ---------------------------------------------------------------------------
# load_source_data — new CSV filenames
# ---------------------------------------------------------------------------


def test_load_source_data_raises_for_missing_price_csv():
    with pytest.raises(FileNotFoundError, match="price_data_rotated_2d.csv"):
        load_source_data(Path("/nonexistent/root"))


def test_load_source_data_raises_for_missing_hourly_csv(tmp_path):
    (tmp_path / "Data").mkdir()
    (tmp_path / "Data" / "price_data_rotated_2d.csv").write_text("timestamp,price\n")

    with pytest.raises(FileNotFoundError, match="historical_hourly_2023_2025.csv"):
        load_source_data(tmp_path)


def test_load_source_data_raises_for_missing_daily_csv(tmp_path):
    (tmp_path / "Data").mkdir()
    (tmp_path / "Data" / "price_data_rotated_2d.csv").write_text("timestamp,price\n")
    (tmp_path / "Data" / "historical_hourly_2023_2025.csv").write_text("timestamp_utc\n")

    with pytest.raises(FileNotFoundError, match="historical_daily_2023_2025.csv"):
        load_source_data(tmp_path)


def test_load_source_data_loads_all_three_csvs(tmp_path):
    (tmp_path / "Data").mkdir()
    (tmp_path / "Data" / "price_data_rotated_2d.csv").write_text("timestamp,price\n2025-01-01,50\n")
    (tmp_path / "Data" / "historical_hourly_2023_2025.csv").write_text(
        "timestamp_utc\n2025-01-01\n"
    )
    (tmp_path / "Data" / "historical_daily_2023_2025.csv").write_text("date_utc\n2025-01-01\n")

    price_df, hourly_df, daily_df = load_source_data(tmp_path)

    assert list(price_df.columns) == ["timestamp", "price"]
    assert "timestamp_utc" in hourly_df.columns
    assert "date_utc" in daily_df.columns


# ---------------------------------------------------------------------------
# preprocess_raw_data
# ---------------------------------------------------------------------------


def test_preprocess_raw_data_returns_dataframe_with_timestamp():
    result = preprocess_raw_data(_price_df(), _hourly_df(), _daily_df())

    assert isinstance(result, pd.DataFrame)
    assert "Timestamp" in result.columns
    assert "price" in result.columns


def test_preprocess_raw_data_drops_missing_price_by_default():
    result = preprocess_raw_data(_price_df(), _hourly_df(), _daily_df(), drop_missing_price=True)

    assert not result["price"].isna().any()


def test_preprocess_raw_data_keeps_missing_price_when_false():
    """drop_missing_price=False is used by the live LGBM pipeline for future rows."""
    price = _price_df()
    # Append a future row with NaN price
    sentinel = pd.DataFrame(
        {"timestamp": [pd.Timestamp("2025-01-01 08:00", tz="UTC")], "price": [np.nan]}
    )
    price_extended = pd.concat([price, sentinel], ignore_index=True)

    result = preprocess_raw_data(
        price_extended, _hourly_df(), _daily_df(), drop_missing_price=False
    )

    assert result["price"].isna().any()


def test_preprocess_raw_data_adds_time_features():
    result = preprocess_raw_data(_price_df(), _hourly_df(), _daily_df())

    for col in ("hour", "day_of_week", "is_weekend", "is_holiday", "tod_sin", "tod_cos"):
        assert col in result.columns, f"Missing time feature: {col}"


def test_preprocess_raw_data_sorted_by_timestamp():
    result = preprocess_raw_data(_price_df(), _hourly_df(), _daily_df())

    ts = pd.to_datetime(result["Timestamp"])
    assert ts.is_monotonic_increasing


def test_preprocess_raw_data_adds_outlier_patching_columns():
    result = preprocess_raw_data(_price_df(), _hourly_df(), _daily_df())

    assert "price_raw" in result.columns
    assert "is_price_patched" in result.columns
