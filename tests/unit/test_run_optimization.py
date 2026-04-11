from __future__ import annotations

from datetime import date

import pandas as pd
import run_optimization as runner_mod
from src.ors.config.optimization_config import (
    BatteryConfiguration,
    OptimizationConfig,
    OptimizationConfiguration,
    OutputConfiguration,
    PVConfiguration,
)


def _make_config() -> OptimizationConfig:
    return OptimizationConfig(
        config_name="runner-test",
        battery=BatteryConfiguration(rated_power_mw=50.0, energy_capacity_mwh=200.0),
        pv=PVConfiguration(
            rated_power_kw=5000.0,
            max_export_kw=4500.0,
            panel_area_m2=25000.0,
            panel_efficiency=0.2,
            generation_source="forecast",
            location_lat=51.5,
            location_lon=-0.12,
        ),
        optimization=OptimizationConfiguration(
            optimization_date=date(2026, 1, 1),
            start_time="00:00",
            duration_hours=1,
            time_step_minutes=15,
            price_source="forecast",
        ),
        output=OutputConfiguration(verbose=False),
    )


def test_runner_uses_merged_forecast_path(monkeypatch):
    config = _make_config()
    runner = runner_mod.OptimizationRunner(config)

    captured: dict[str, object] = {}

    def fake_create_input_df(**kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2026-01-01 00:00:00", periods=4, freq="15min", tz="UTC"
                ),
                "price_intraday": [50.0, 55.0, 60.0, 65.0],
                "generation_kw": [1000.0, 2000.0, 3000.0, 4000.0],
            }
        )

    monkeypatch.setattr(runner_mod, "create_input_df", fake_create_input_df)
    monkeypatch.setattr(
        runner_mod,
        "load_price_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    monkeypatch.setattr(
        runner_mod,
        "load_solar_data",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    price_data, terminal_price, solar_data = runner._load_data()

    assert captured["config"] is config.pv
    assert captured["time_step_minutes"] == 15
    assert price_data == {1: 50.0, 2: 55.0, 3: 60.0, 4: 65.0}
    assert solar_data == {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}
    assert terminal_price == 57.5


def test_runner_falls_back_to_independent_loaders_when_merge_fails(monkeypatch):
    config = _make_config()
    runner = runner_mod.OptimizationRunner(config)

    monkeypatch.setattr(
        runner_mod,
        "create_input_df",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("merge failed")),
    )
    monkeypatch.setattr(
        runner_mod,
        "load_price_data",
        lambda *args, **kwargs: ({1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}, 25.0),
    )
    monkeypatch.setattr(
        runner_mod,
        "load_solar_data",
        lambda *args, **kwargs: {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4},
    )

    price_data, terminal_price, solar_data = runner._load_data()

    assert price_data == {1: 10.0, 2: 20.0, 3: 30.0, 4: 40.0}
    assert terminal_price == 25.0
    assert solar_data == {1: 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
