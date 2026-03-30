from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import src.ors.services.optimizer.integration as m
from src.ors.config.optimization_config import PVConfiguration


def _dt_range(start: datetime, periods: int, freq: str = "15min") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")


@pytest.fixture
def runtime_pv_config() -> PVConfiguration:
    return PVConfiguration(
        rated_power_kw=5000.0,
        max_export_kw=4500.0,
        panel_area_m2=25000.0,
        panel_efficiency=0.2,
        generation_source="forecast",
        location_lat=51.5,
        location_lon=-0.12,
    )


def test_create_input_df_aligns_to_requested_horizon(monkeypatch, runtime_pv_config):
    start = datetime(2026, 3, 1, 10, 7, tzinfo=timezone.utc)
    expected_start = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    end = expected_start + timedelta(hours=1)

    captured: dict[str, object] = {}

    def fake_generate_pv_df(config, *, client, start_datetime, end_datetime, time_step_minutes):
        captured["start_datetime"] = start_datetime
        captured["end_datetime"] = end_datetime
        captured["time_step_minutes"] = time_step_minutes
        assert config is runtime_pv_config
        return pd.DataFrame(
            {
                "timestamp": _dt_range(expected_start, periods=4),
                "generation_kw": [10.0, 20.0, 30.0, 40.0],
            }
        )

    def fake_get_price_forecast_df(start_datetime, end_datetime, time_step_minutes, **kwargs):
        assert start_datetime == expected_start
        assert end_datetime == end
        assert time_step_minutes == 15
        return pd.DataFrame(
            {
                "timestamp": _dt_range(expected_start, periods=4),
                "price_intraday": [100.0, 101.0, 102.0, 103.0],
            }
        )

    monkeypatch.setattr(m, "_generate_pv_df", fake_generate_pv_df)
    monkeypatch.setattr(m, "get_price_forecast_df", fake_get_price_forecast_df)

    df = m.create_input_df(
        config=runtime_pv_config,
        start_datetime=start,
        end_datetime=end,
    )

    assert captured["start_datetime"] == expected_start
    assert captured["end_datetime"] == end
    assert captured["time_step_minutes"] == 15
    assert list(df.columns) == ["timestamp", "generation_kw", "price_intraday"]
    assert list(pd.to_datetime(df["timestamp"], utc=True)) == list(_dt_range(expected_start, 4))
    assert df["generation_kw"].tolist() == pytest.approx([10.0, 20.0, 30.0, 40.0])
    assert df["price_intraday"].tolist() == pytest.approx([100.0, 101.0, 102.0, 103.0])


def test_create_input_df_supports_30_minute_resolution(monkeypatch, runtime_pv_config):
    start = datetime(2026, 3, 1, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=2)

    def fake_generate_pv_df(config, *, client, start_datetime, end_datetime, time_step_minutes):
        assert time_step_minutes == 30
        return pd.DataFrame(
            {
                "timestamp": _dt_range(start, periods=4, freq="30min"),
                "generation_kw": [10.0, 20.0, 30.0, 40.0],
            }
        )

    def fake_get_price_forecast_df(start_datetime, end_datetime, time_step_minutes, **kwargs):
        assert time_step_minutes == 30
        return pd.DataFrame(
            {
                "timestamp": _dt_range(start, periods=4, freq="30min"),
                "price_intraday": [50.0, 55.0, 60.0, 65.0],
            }
        )

    monkeypatch.setattr(m, "_generate_pv_df", fake_generate_pv_df)
    monkeypatch.setattr(m, "get_price_forecast_df", fake_get_price_forecast_df)

    df = m.create_input_df(
        config=runtime_pv_config,
        start_datetime=start,
        end_datetime=end,
        time_step_minutes=30,
    )

    assert len(df) == 4
    assert df["timestamp"].iloc[0] == pd.Timestamp(start)
    assert df["price_intraday"].tolist() == pytest.approx([50.0, 55.0, 60.0, 65.0])


def test_create_input_df_raises_if_price_horizon_has_gaps(monkeypatch, runtime_pv_config):
    start = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)

    def fake_generate_pv_df(config, *, client, start_datetime, end_datetime, time_step_minutes):
        return pd.DataFrame(
            {
                "timestamp": _dt_range(start, periods=4),
                "generation_kw": [1.0, 2.0, 3.0, 4.0],
            }
        )

    def fake_get_price_forecast_df(start_datetime, end_datetime, time_step_minutes, **kwargs):
        return pd.DataFrame(
            {
                "timestamp": _dt_range(start, periods=3),
                "price_intraday": [100.0, 101.0, 102.0],
            }
        )

    monkeypatch.setattr(m, "_generate_pv_df", fake_generate_pv_df)
    monkeypatch.setattr(m, "get_price_forecast_df", fake_get_price_forecast_df)

    with pytest.raises(ValueError, match="requested optimization horizon"):
        m.create_input_df(
            config=runtime_pv_config,
            start_datetime=start,
            end_datetime=end,
        )


def test_create_input_df_passes_custom_model_path(monkeypatch, runtime_pv_config):
    start = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    captured: dict[str, object] = {}

    def fake_generate_pv_df(config, *, client, start_datetime, end_datetime, time_step_minutes):
        return pd.DataFrame(
            {
                "timestamp": _dt_range(start, periods=4),
                "generation_kw": [1.0, 2.0, 3.0, 4.0],
            }
        )

    def fake_get_price_forecast_df(start_datetime, end_datetime, time_step_minutes, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "timestamp": _dt_range(start, periods=4),
                "price_intraday": [1.0, 2.0, 3.0, 4.0],
            }
        )

    monkeypatch.setattr(m, "_generate_pv_df", fake_generate_pv_df)
    monkeypatch.setattr(m, "get_price_forecast_df", fake_get_price_forecast_df)

    custom_path = Path("/custom/model/dir")
    m.create_input_df(
        config=runtime_pv_config,
        start_datetime=start,
        end_datetime=end,
        model_path=custom_path,
    )

    assert captured["model_path"] == custom_path
