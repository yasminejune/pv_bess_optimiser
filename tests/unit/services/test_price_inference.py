"""Unit tests for the price inference service.

Covers:
- live_inference.py: prepare_features_for_inference, load_model, select_forecast_rows
- live_data_pipeline.py: build_live_merged_dataset (via mocked helpers)
- reference_time: historical replay mode for both select_forecast_rows and
  build_live_merged_dataset
"""

from __future__ import annotations

import pickle
from datetime import datetime, timedelta, timezone
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
    now = (
        pd.Timestamp.utcnow().tz_convert(None).floor("h")
    )  # tz-naive UTC, matches function internals
    timestamps = [now + pd.Timedelta(hours=i) for i in range(10)]
    df = pd.DataFrame({"Timestamp": timestamps, "val": range(10)})

    out = li.select_forecast_rows(df, horizon_hours=3)

    assert list(out["val"]) == [0, 1, 2]


def test_select_forecast_rows_returns_copy():
    now = (
        pd.Timestamp.utcnow().tz_convert(None).floor("h")
    )  # tz-naive UTC, matches function internals
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
    now = (
        pd.Timestamp.utcnow().tz_convert(None).floor("h")
    )  # tz-naive UTC, matches function internals
    timestamps = [now + pd.Timedelta(hours=i) for i in range(3)]
    df = pd.DataFrame({"Timestamp": timestamps, "val": range(3)})

    out = li.select_forecast_rows(df, horizon_hours=10)

    assert len(out) == 3


def test_select_forecast_rows_reference_time_used_as_anchor():
    ref = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
    ref_ts = pd.Timestamp(ref).tz_convert(None).floor("h")  # tz-naive, matches production
    timestamps = [ref_ts + pd.Timedelta(hours=i) for i in range(10)]
    df = pd.DataFrame({"Timestamp": timestamps, "val": range(10)})

    out = li.select_forecast_rows(df, horizon_hours=3, reference_time=ref)

    assert list(out["val"]) == [0, 1, 2]
    assert out["Timestamp"].iloc[0] == ref_ts


def test_select_forecast_rows_reference_time_excludes_earlier_rows():
    ref = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
    ref_ts = pd.Timestamp(ref).tz_convert(None).floor("h")  # tz-naive, matches production
    # rows before and after reference_time
    timestamps = [ref_ts - pd.Timedelta(hours=2), ref_ts, ref_ts + pd.Timedelta(hours=1)]
    df = pd.DataFrame({"Timestamp": timestamps, "val": [10, 20, 30]})

    out = li.select_forecast_rows(df, horizon_hours=5, reference_time=ref)

    assert 10 not in list(out["val"])
    assert list(out["val"]) == [20, 30]


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
    monkeypatch.setattr(
        ldp, "_fetch_live_weather", lambda **k: (_make_hourly_df(), _make_daily_df())
    )
    monkeypatch.setattr(ldp, "_fetch_live_price", lambda **k: _make_price_df())

    result = ldp.build_live_merged_dataset(lag_steps=(1, 2))

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert "Timestamp" in result.columns


def test_build_live_merged_dataset_sorted_by_timestamp(monkeypatch):
    monkeypatch.setattr(
        ldp, "_fetch_live_weather", lambda **k: (_make_hourly_df(), _make_daily_df())
    )
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
    monkeypatch.setattr(
        ldp, "_fetch_live_weather", lambda **k: (_make_hourly_df(), _make_daily_df())
    )
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


def test_build_live_merged_dataset_passes_reference_time(monkeypatch):
    ref = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
    captured: dict = {}

    def fake_weather(**k):
        captured["weather_ref"] = k.get("reference_time")
        return _make_hourly_df(), _make_daily_df()

    def fake_price(**k):
        captured["price_ref"] = k.get("reference_time")
        return _make_price_df()

    monkeypatch.setattr(ldp, "_fetch_live_weather", fake_weather)
    monkeypatch.setattr(ldp, "_fetch_live_price", fake_price)

    ldp.build_live_merged_dataset(lag_steps=(1,), reference_time=ref)

    assert captured["weather_ref"] == ref
    assert captured["price_ref"] == ref


# ---------------------------------------------------------------------------
# find_latest_lgbm_model
# ---------------------------------------------------------------------------


def test_find_latest_lgbm_model_missing_dir_raises(tmp_path):
    missing = tmp_path / "nonexistent_dir"
    with pytest.raises(FileNotFoundError, match="LGBM model directory not found"):
        li.find_latest_lgbm_model(missing)


def test_find_latest_lgbm_model_empty_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="No LGBM model files found"):
        li.find_latest_lgbm_model(tmp_path)


def test_find_latest_lgbm_model_returns_most_recent(tmp_path):
    (tmp_path / "recursive_single_model_20260101_100000.joblib").touch()
    newer = tmp_path / "recursive_single_model_20260101_110000.joblib"
    newer.touch()
    meta = tmp_path / "recursive_single_model_20260101_110000_meta.joblib"
    meta.touch()

    model_path, meta_path = li.find_latest_lgbm_model(tmp_path)

    assert model_path == newer
    assert meta_path == meta


def test_find_latest_lgbm_model_no_meta_returns_none(tmp_path):
    (tmp_path / "recursive_single_model_20260101_100000.joblib").touch()

    _, meta_path = li.find_latest_lgbm_model(tmp_path)

    assert meta_path is None


def test_find_latest_lgbm_model_excludes_meta_files(tmp_path):
    model = tmp_path / "recursive_single_model_20260101_100000.joblib"
    model.touch()
    (tmp_path / "recursive_single_model_20260101_100000_meta.joblib").touch()

    model_path, _ = li.find_latest_lgbm_model(tmp_path)

    assert model_path == model
    assert "_meta" not in model_path.stem


# ---------------------------------------------------------------------------
# load_model — directory dispatch
# ---------------------------------------------------------------------------


def test_load_model_with_directory_delegates_to_find_latest(tmp_path, monkeypatch):
    """load_model(dir) calls find_latest_lgbm_model and loads the resolved file."""
    model_obj = SimpleNamespace(name="lgbm_dummy")
    pkl_path = tmp_path / "lgbm_model.pkl"
    with open(pkl_path, "wb") as fh:
        pickle.dump(model_obj, fh)

    # Redirect find_latest_lgbm_model to our pkl file to avoid joblib dependency
    monkeypatch.setattr(li, "find_latest_lgbm_model", lambda d: (pkl_path, None))

    loaded = li.load_model(tmp_path)

    assert loaded.name == "lgbm_dummy"


# ---------------------------------------------------------------------------
# _is_lgbm_forecaster
# ---------------------------------------------------------------------------


def test_is_lgbm_forecaster_true_when_class_name_matches():
    class ForecasterRecursive:
        pass

    assert li._is_lgbm_forecaster(ForecasterRecursive()) is True


def test_is_lgbm_forecaster_false_for_other_types():
    assert li._is_lgbm_forecaster(object()) is False
    assert li._is_lgbm_forecaster(SimpleNamespace()) is False


# ---------------------------------------------------------------------------
# _extract_lgbm_inputs
# ---------------------------------------------------------------------------


def test_extract_lgbm_inputs_returns_correct_shapes():
    ref = datetime(2025, 6, 1, 0, 0, tzinfo=timezone.utc)
    ref_naive = pd.Timestamp(ref).tz_convert(None).floor("15min")

    past_times = [ref_naive - pd.Timedelta(minutes=15 * (39 - i)) for i in range(40)]
    future_times = pd.date_range(ref_naive, periods=4, freq="15min")

    df = pd.concat(
        [
            pd.DataFrame({"Timestamp": past_times, "price": np.ones(40) * 50.0, "f1": np.ones(40)}),
            pd.DataFrame({"Timestamp": future_times, "price": [np.nan] * 4, "f1": np.ones(4)}),
        ],
        ignore_index=True,
    )

    window_size = 10
    last_window, exog, forecast_index = li._extract_lgbm_inputs(
        df, feature_cols=["f1"], window_size=window_size, forecast_steps=4, reference_time=ref
    )

    assert len(last_window) == window_size
    assert len(exog) == 4
    assert len(forecast_index) == 4


def test_extract_lgbm_inputs_exog_index_starts_at_window_size():
    """exog.index[0] must equal window_size — required by skforecast."""
    ref = datetime(2025, 6, 1, 0, 0, tzinfo=timezone.utc)
    ref_naive = pd.Timestamp(ref).tz_convert(None).floor("15min")

    past_times = [ref_naive - pd.Timedelta(minutes=15 * (19 - i)) for i in range(20)]
    future_times = pd.date_range(ref_naive, periods=3, freq="15min")

    df = pd.concat(
        [
            pd.DataFrame(
                {"Timestamp": past_times, "price": np.ones(20) * 100.0, "f1": np.ones(20)}
            ),
            pd.DataFrame({"Timestamp": future_times, "price": [np.nan] * 3, "f1": np.ones(3)}),
        ],
        ignore_index=True,
    )

    window_size = 8
    _, exog, _ = li._extract_lgbm_inputs(
        df, feature_cols=["f1"], window_size=window_size, forecast_steps=3, reference_time=ref
    )

    assert int(exog.index[0]) == window_size


# ---------------------------------------------------------------------------
# live_data_pipeline — _chunked_fetch
# ---------------------------------------------------------------------------


def test_chunked_fetch_makes_multiple_calls_for_wide_window():
    calls = []

    def fake_fetch(start, end):
        calls.append((start, end))
        return pd.DataFrame({"ts_utc": [start], "value": [1.0]})

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=400)  # > 168 h limit

    result = ldp._chunked_fetch(fake_fetch, start, end, max_hours=168)

    assert len(calls) > 1
    assert isinstance(result, pd.DataFrame)
    assert len(result) > 0


def test_chunked_fetch_single_call_within_limit():
    calls = []

    def fake_fetch(start, end):
        calls.append((start, end))
        return pd.DataFrame({"ts_utc": [start], "value": [1.0]})

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=48)  # < 168 h

    ldp._chunked_fetch(fake_fetch, start, end, max_hours=168)

    assert len(calls) == 1


def test_chunked_fetch_deduplicates_on_ts_utc():
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)

    def fake_fetch(start, end):
        return pd.DataFrame({"ts_utc": [ts], "value": [1.0]})

    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    end = start + timedelta(hours=400)

    result = ldp._chunked_fetch(fake_fetch, start, end, max_hours=168)

    # Multiple chunks returned the same ts_utc; dedup should leave one row
    assert result["ts_utc"].nunique() == 1


# ---------------------------------------------------------------------------
# live_data_pipeline — build_live_lgbm_dataset
# ---------------------------------------------------------------------------


def test_build_live_lgbm_dataset_returns_dataframe(monkeypatch):
    def fake_weather(**k):
        return _make_hourly_df(200), _make_daily_df(10)

    def fake_price(**k):
        return _make_price_df(200)

    def fake_preprocess(price_df, weather_df, daily_df, drop_missing_price=True):
        return pd.DataFrame(
            {"Timestamp": pd.date_range("2025-01-01", periods=5, freq="15min"), "price": np.ones(5)}
        )

    monkeypatch.setattr(ldp, "_fetch_live_weather", fake_weather)
    monkeypatch.setattr(ldp, "_fetch_live_price", fake_price)
    monkeypatch.setattr(ldp, "preprocess_raw_data", fake_preprocess)

    result = ldp.build_live_lgbm_dataset(forecast_steps=4)

    assert isinstance(result, pd.DataFrame)
    assert "Timestamp" in result.columns


def test_build_live_lgbm_dataset_raises_on_empty_price(monkeypatch):
    def fake_weather(**k):
        return _make_hourly_df(200), _make_daily_df(10)

    def fake_price(**k):
        return pd.DataFrame(columns=["timestamp", "price"])

    monkeypatch.setattr(ldp, "_fetch_live_weather", fake_weather)
    monkeypatch.setattr(ldp, "_fetch_live_price", fake_price)

    with pytest.raises(ValueError, match="Price API returned no data"):
        ldp.build_live_lgbm_dataset(forecast_steps=4)


def test_build_live_lgbm_dataset_sorted_by_timestamp(monkeypatch):
    def fake_weather(**k):
        return _make_hourly_df(200), _make_daily_df(10)

    def fake_price(**k):
        return _make_price_df(200)

    unsorted_times = pd.date_range("2025-01-01", periods=5, freq="15min")[::-1]

    def fake_preprocess(price_df, weather_df, daily_df, drop_missing_price=True):
        return pd.DataFrame({"Timestamp": unsorted_times, "price": np.ones(5)})

    monkeypatch.setattr(ldp, "_fetch_live_weather", fake_weather)
    monkeypatch.setattr(ldp, "_fetch_live_price", fake_price)
    monkeypatch.setattr(ldp, "preprocess_raw_data", fake_preprocess)

    result = ldp.build_live_lgbm_dataset(forecast_steps=4)

    ts = pd.to_datetime(result["Timestamp"])
    assert ts.is_monotonic_increasing


def test_build_live_lgbm_dataset_passes_reference_time(monkeypatch):
    ref = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
    captured: dict = {}

    def fake_weather(**k):
        captured["weather_ref"] = k.get("reference_time")
        return _make_hourly_df(200), _make_daily_df(10)

    def fake_price(**k):
        captured["price_ref"] = k.get("reference_time")
        return _make_price_df(200)

    def fake_preprocess(price_df, weather_df, daily_df, drop_missing_price=True):
        return pd.DataFrame(
            {"Timestamp": pd.date_range("2025-01-01", periods=3, freq="15min"), "price": np.ones(3)}
        )

    monkeypatch.setattr(ldp, "_fetch_live_weather", fake_weather)
    monkeypatch.setattr(ldp, "_fetch_live_price", fake_price)
    monkeypatch.setattr(ldp, "preprocess_raw_data", fake_preprocess)

    ldp.build_live_lgbm_dataset(forecast_steps=4, reference_time=ref)

    assert captured["weather_ref"] == ref
    assert captured["price_ref"] == ref
