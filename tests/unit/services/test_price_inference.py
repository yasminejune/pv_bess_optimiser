"""Unit tests for the price inference service.

Covers:
- live_inference.py: prepare_features_for_inference, load_model, select_forecast_rows
- live_data_pipeline.py: build_live_merged_dataset (via mocked helpers)
"""

from __future__ import annotations

import pickle
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import src.ors.services.price_inference.live_data_pipeline as ldp
import src.ors.services.price_inference.live_inference as li

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(feature_names: list[str]) -> MagicMock:
    """Return a mock model with feature_names_in_ set."""
    model = MagicMock()
    model.feature_names_in_ = np.array(feature_names)
    model.predict.return_value = np.ones(len(feature_names))
    return model


def _make_df(cols: list[str], n: int = 3) -> pd.DataFrame:
    """Return a simple numeric DataFrame."""
    return pd.DataFrame({c: np.arange(float(n)) for c in cols})


# ---------------------------------------------------------------------------
# prepare_features_for_inference
# ---------------------------------------------------------------------------


def test_prepare_features_drops_target_and_timestamp():
    df = _make_df(["Price", "Timestamp", "feat_a", "feat_b"])
    df["Timestamp"] = pd.date_range("2025-01-01", periods=3, freq="h")
    model = _make_model(["feat_a", "feat_b"])

    out = li.prepare_features_for_inference(df, model, target_col="Price")

    assert "Price" not in out.columns
    assert "Timestamp" not in out.columns
    assert list(out.columns) == ["feat_a", "feat_b"]


def test_prepare_features_casts_bool_to_int():
    df = pd.DataFrame({"flag": [True, False, True], "val": [1.0, 2.0, 3.0]})
    model = _make_model(["flag", "val"])

    out = li.prepare_features_for_inference(df, model)

    assert out["flag"].dtype != bool
    assert set(out["flag"].unique()).issubset({0, 1})


def test_prepare_features_fills_nans_with_median():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
    model = _make_model(["a"])

    out = li.prepare_features_for_inference(df, model)

    assert not out["a"].isna().any()
    assert out["a"].iloc[1] == pytest.approx(2.0)  # median of [1, 3]


def test_prepare_features_reindex_adds_missing_as_zero():
    df = _make_df(["feat_a"])
    model = _make_model(["feat_a", "feat_missing"])

    out = li.prepare_features_for_inference(df, model)

    assert "feat_missing" in out.columns
    assert (out["feat_missing"] == 0).all()


def test_prepare_features_reindex_drops_extra_columns():
    df = _make_df(["feat_a", "extra_col"])
    model = _make_model(["feat_a"])

    out = li.prepare_features_for_inference(df, model)

    assert "extra_col" not in out.columns
    assert list(out.columns) == ["feat_a"]


def test_prepare_features_output_is_numeric():
    df = pd.DataFrame({"a": [True, False], "b": [1.0, 2.0]})
    model = _make_model(["a", "b"])

    out = li.prepare_features_for_inference(df, model)

    for col in out.columns:
        assert pd.api.types.is_numeric_dtype(out[col])


def test_prepare_features_column_order_matches_model():
    df = _make_df(["z_feat", "a_feat"])
    model = _make_model(["a_feat", "z_feat"])

    out = li.prepare_features_for_inference(df, model)

    assert list(out.columns) == ["a_feat", "z_feat"]


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


def test_load_model_raises_file_not_found(tmp_path):
    missing = tmp_path / "no_model.pkl"

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        li.load_model(missing)


def test_load_model_returns_object(tmp_path):
    model_obj = SimpleNamespace(name="dummy_model")
    pkl_path = tmp_path / "model.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(model_obj, fh)

    loaded = li.load_model(pkl_path)

    assert loaded.name == "dummy_model"


# ---------------------------------------------------------------------------
# select_forecast_rows
# ---------------------------------------------------------------------------


def test_select_forecast_rows_returns_last_n():
    now = pd.Timestamp.now().floor("h")
    timestamps = [now + pd.Timedelta(hours=i) for i in range(10)]
    df = pd.DataFrame({"Timestamp": timestamps, "val": range(10)})

    out = li.select_forecast_rows(df, horizon_hours=3)

    assert list(out["val"]) == [0, 1, 2]


def test_select_forecast_rows_returns_copy():
    now = pd.Timestamp.now().floor("h")
    timestamps = [now + pd.Timedelta(hours=i) for i in range(5)]
    df = pd.DataFrame({"Timestamp": timestamps, "val": range(5)})
    out = li.select_forecast_rows(df, horizon_hours=2)
    out["val"] = 99

    assert df["val"].iloc[0] != 99


def test_select_forecast_rows_empty_df_prints_warning(capsys):
    out = li.select_forecast_rows(pd.DataFrame(columns=["Timestamp", "val"]), horizon_hours=5)

    assert out.empty
    assert "Warning" in capsys.readouterr().out


def test_select_forecast_rows_horizon_larger_than_df():
    now = pd.Timestamp.now().floor("h")
    timestamps = [now + pd.Timedelta(hours=i) for i in range(3)]
    df = pd.DataFrame({"Timestamp": timestamps, "val": range(3)})

    out = li.select_forecast_rows(df, horizon_hours=10)

    assert len(out) == 3


# ---------------------------------------------------------------------------
# live_data_pipeline — build_live_merged_dataset
# ---------------------------------------------------------------------------


def _make_hourly_df(n: int = 72) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2025-01-01", periods=n, freq="h"),
            "temperature_2m": np.ones(n),
            "relative_humidity_2m": np.ones(n),
            "dew_point_2m": np.ones(n),
            "precipitation": np.zeros(n),
            "rain": np.zeros(n),
            "snowfall": np.zeros(n),
            "cloud_cover": np.zeros(n),
            "cloud_cover_low": np.zeros(n),
            "cloud_cover_mid": np.zeros(n),
            "cloud_cover_high": np.zeros(n),
            "is_day": np.ones(n),
            "shortwave_radiation": np.ones(n),
            "direct_radiation": np.ones(n),
            "wind_speed_10m": np.ones(n),
            "wind_gusts_10m": np.ones(n),
            "wind_direction_10m": np.ones(n),
            "surface_pressure": np.ones(n) * 1013,
            "weather_code": np.zeros(n),
        }
    )


def _make_daily_df(n: int = 3) -> pd.DataFrame:
    base = pd.date_range("2025-01-01", periods=n, freq="D")
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
            "temperature_2m_max": np.ones(n) * 15,
            "temperature_2m_min": np.ones(n) * 5,
        }
    )


def _make_price_df(n: int = 48) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="h"),
            "price": np.random.uniform(50, 150, n),
            "demand_itsdo": np.random.uniform(30000, 40000, n),
            "demand_indo": np.random.uniform(30000, 40000, n),
            "demand_inddem": np.random.uniform(30000, 40000, n),
        }
    )


def test_build_live_merged_dataset_returns_dataframe(monkeypatch):
    monkeypatch.setattr(ldp, "_fetch_live_weather", lambda **k: (_make_hourly_df(), _make_daily_df()))
    monkeypatch.setattr(ldp, "_fetch_live_price", lambda **k: _make_price_df())

    result = ldp.build_live_merged_dataset(lag_steps=(1, 2))

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "Timestamp" in result.columns


def test_build_live_merged_dataset_sorted_by_timestamp(monkeypatch):
    monkeypatch.setattr(ldp, "_fetch_live_weather", lambda **k: (_make_hourly_df(), _make_daily_df()))
    monkeypatch.setattr(ldp, "_fetch_live_price", lambda **k: _make_price_df())

    result = ldp.build_live_merged_dataset(lag_steps=(1,))

    ts = pd.to_datetime(result["Timestamp"])
    assert ts.is_monotonic_increasing


def test_build_live_merged_dataset_raises_on_no_overlap(monkeypatch):
    hourly = _make_hourly_df()
    hourly["timestamp_utc"] = pd.date_range("2025-06-01", periods=len(hourly), freq="h")

    daily = _make_daily_df()
    # daily stays at 2025-01-01 — no overlap with hourly

    monkeypatch.setattr(ldp, "_fetch_live_weather", lambda **k: (hourly, daily))
    monkeypatch.setattr(ldp, "_fetch_live_price", lambda **k: _make_price_df())

    with pytest.raises(ValueError, match="No overlapping timestamp window"):
        ldp.build_live_merged_dataset(lag_steps=(1,))


def test_build_live_merged_dataset_all_price_endpoints_unavailable(monkeypatch):
    monkeypatch.setattr(ldp, "_fetch_live_weather", lambda **k: (_make_hourly_df(), _make_daily_df()))
    monkeypatch.setattr(
        ldp,
        "_fetch_live_price",
        lambda **k: pd.DataFrame(
            columns=["timestamp", "price", "demand_itsdo", "demand_indo", "demand_inddem"]
        ),
    )

    result = ldp.build_live_merged_dataset(lag_steps=(1,))

    assert isinstance(result, pd.DataFrame)
    assert "Timestamp" in result.columns
