# tests/test_create_input_df.py

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

# TODO: change this import to your real module
# e.g. from ors.services.input_builder import create_input_df
import src.ors.services.optimizer.integration as m


def _dt_range_15m(start: datetime, periods: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq="15min", tz="UTC")


@pytest.fixture
def config_stub():
    # create_input_df only passes config through; it doesn't inspect fields.
    return object()


def test_create_input_df_outer_merge_union_and_timestamp_handling(monkeypatch, config_stub):
    """
    Verifies:
      - PV already has 'timestamp_utc'
      - Price has 'Timestamp' and gets renamed to 'timestamp_utc' for merge
      - Outer merge preserves union of timestamps
      - Output sorted by timestamp_utc
    """
    start = datetime(2026, 3, 1, 10, 7, tzinfo=timezone.utc)  # not on boundary
    expected_start = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)  # floored
    end = expected_start + timedelta(hours=1)

    pv_times = _dt_range_15m(expected_start, periods=5)  # 10:00..11:00
    price_times = _dt_range_15m(expected_start + timedelta(minutes=15), periods=4)  # 10:15..11:00

    def fake_generate_pv_power_for_date_range(*, config, start_datetime, end_datetime, **kwargs):
        assert start_datetime == expected_start
        assert end_datetime == end
        return pd.DataFrame(
            {
                "timestamp_utc": pv_times,
                "generation_kw": [0, 1, 2, 3, 4],
            }
        )

    def fake_run_inference(**kwargs):
        # Price has Timestamp (NOT timestamp_utc)
        return pd.DataFrame(
            {
                "Timestamp": price_times,
                "price": [100.0, 101.0, 102.0, 103.0],
            }
        )

    monkeypatch.setattr(
        m, "generate_pv_power_for_date_range", fake_generate_pv_power_for_date_range
    )
    monkeypatch.setattr(m, "run_inference", fake_run_inference)

    df = m.create_input_df(
        config=config_stub,
        start_datetime=start,
        end_datetime=end,
    )

    assert "timestamp" in df.columns
    assert "generation_kw" in df.columns
    assert "price" in df.columns

    expected_union = pv_times.union(price_times)
    got_times = pd.to_datetime(df["timestamp"], utc=True)
    assert got_times.is_monotonic_increasing
    assert list(got_times) == list(expected_union)

    # Outer join NaNs expected:
    # 10:00 exists only in PV -> price NaN
    first_row = df.iloc[0]
    assert pd.isna(first_row["price"])
    assert first_row["generation_kw"] == 0

    # 10:15 exists in both -> not NaN on either
    row_1015 = df[df["timestamp"] == pd.Timestamp("2026-03-01T10:15:00Z")].iloc[0]
    assert row_1015["generation_kw"] == 1
    assert row_1015["price"] == 100.0


def test_create_input_df_injects_horizon_hours_when_missing(monkeypatch, config_stub):
    """
    Verifies horizon_hours is injected into kwargs only if missing.
    """
    start = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=24)
    expected_horizon = 24.0

    def fake_generate_pv_power_for_date_range(*, config, start_datetime, end_datetime, **kwargs):
        return pd.DataFrame(
            {"timestamp_utc": _dt_range_15m(start, periods=2), "generation_kw": [0, 1]}
        )

    captured = {}

    def fake_run_inference(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"Timestamp": _dt_range_15m(start, periods=2), "price": [1.0, 2.0]})

    monkeypatch.setattr(
        m, "generate_pv_power_for_date_range", fake_generate_pv_power_for_date_range
    )
    monkeypatch.setattr(m, "run_inference", fake_run_inference)

    _ = m.create_input_df(config=config_stub, start_datetime=start, end_datetime=end)

    assert "horizon_hours" in captured
    assert captured["horizon_hours"] == pytest.approx(expected_horizon, rel=1e-9)


def test_create_input_df_does_not_override_horizon_hours_if_provided(monkeypatch, config_stub):
    """
    Verifies user-provided horizon_hours is respected and not overwritten.
    """
    start = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=24)

    def fake_generate_pv_power_for_date_range(*, config, start_datetime, end_datetime, **kwargs):
        return pd.DataFrame(
            {"timestamp_utc": _dt_range_15m(start, periods=1), "generation_kw": [0]}
        )

    captured = {}

    def fake_run_inference(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"Timestamp": _dt_range_15m(start, periods=1), "price": [1.0]})

    monkeypatch.setattr(
        m, "generate_pv_power_for_date_range", fake_generate_pv_power_for_date_range
    )
    monkeypatch.setattr(m, "run_inference", fake_run_inference)

    _ = m.create_input_df(
        config=config_stub,
        start_datetime=start,
        end_datetime=end,
        horizon_hours=999.0,  # should NOT be overwritten
    )

    assert captured["horizon_hours"] == 999.0


def test_raises_if_pv_missing_timestamp_utc_column(monkeypatch, config_stub):
    def fake_generate_pv_power_for_date_range(*, config, start_datetime, end_datetime, **kwargs):
        return pd.DataFrame({"NOT_TIMESTAMP": [], "generation_kw": []})

    def fake_run_inference(**kwargs):
        return pd.DataFrame({"Timestamp": [], "price": []})

    monkeypatch.setattr(
        m, "generate_pv_power_for_date_range", fake_generate_pv_power_for_date_range
    )
    monkeypatch.setattr(m, "run_inference", fake_run_inference)

    with pytest.raises(KeyError, match="timestamp_utc"):
        _ = m.create_input_df(config=config_stub)


def test_raises_if_price_missing_timestamp_column(monkeypatch, config_stub):
    start = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)

    def fake_generate_pv_power_for_date_range(*, config, start_datetime, end_datetime, **kwargs):
        return pd.DataFrame(
            {"timestamp_utc": _dt_range_15m(start, periods=1), "generation_kw": [0]}
        )

    def fake_run_inference(**kwargs):
        return pd.DataFrame({"NOT_TIMESTAMP": [], "price": []})

    monkeypatch.setattr(
        m, "generate_pv_power_for_date_range", fake_generate_pv_power_for_date_range
    )
    monkeypatch.setattr(m, "run_inference", fake_run_inference)

    with pytest.raises(KeyError, match="Timestamp"):
        _ = m.create_input_df(config=config_stub)


def test_create_input_df_uses_lgbm_model_dir_by_default(monkeypatch, config_stub):
    """Verifies run_inference receives model_path=LGBM_MODEL_DIR when none is passed."""
    from src.ors.services.price_inference.live_inference import LGBM_MODEL_DIR

    start = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=24)
    captured: dict = {}

    def fake_generate_pv_power_for_date_range(*, config, start_datetime, end_datetime, **kwargs):
        return pd.DataFrame(
            {"timestamp_utc": _dt_range_15m(start, periods=1), "generation_kw": [0]}
        )

    def fake_run_inference(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"Timestamp": _dt_range_15m(start, periods=1), "Price_pred": [1.0]})

    monkeypatch.setattr(
        m, "generate_pv_power_for_date_range", fake_generate_pv_power_for_date_range
    )
    monkeypatch.setattr(m, "run_inference", fake_run_inference)

    _ = m.create_input_df(config=config_stub, start_datetime=start, end_datetime=end)

    assert "model_path" in captured
    assert captured["model_path"] == LGBM_MODEL_DIR


def test_create_input_df_passes_custom_model_path_to_run_inference(monkeypatch, config_stub):
    """Verifies an explicit model_path is forwarded unchanged to run_inference."""
    from pathlib import Path

    start = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=24)
    custom_path = Path("/custom/model/dir")
    captured: dict = {}

    def fake_generate_pv_power_for_date_range(*, config, start_datetime, end_datetime, **kwargs):
        return pd.DataFrame(
            {"timestamp_utc": _dt_range_15m(start, periods=1), "generation_kw": [0]}
        )

    def fake_run_inference(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame({"Timestamp": _dt_range_15m(start, periods=1), "Price_pred": [1.0]})

    monkeypatch.setattr(
        m, "generate_pv_power_for_date_range", fake_generate_pv_power_for_date_range
    )
    monkeypatch.setattr(m, "run_inference", fake_run_inference)

    _ = m.create_input_df(
        config=config_stub,
        start_datetime=start,
        end_datetime=end,
        model_path=custom_path,
    )

    assert captured["model_path"] == custom_path
