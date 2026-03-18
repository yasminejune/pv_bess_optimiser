"""Tests for optimization_config module."""

from datetime import date, datetime

import pytest
from pydantic import ValidationError
from src.ors.config.optimization_config import (
    BatteryConfiguration,
    OptimizationConfig,
    OptimizationConfiguration,
    OutputConfiguration,
    PVConfiguration,
    load_config_from_json,
    save_config_to_json,
)

# ---------------------------------------------------------------------------
# PVConfiguration
# ---------------------------------------------------------------------------


class TestPVConfiguration:
    def test_valid_pv_config(self):
        pv = PVConfiguration(
            rated_power_kw=100.0,
            generation_source="forecast",
        )
        assert pv.rated_power_kw == 100.0
        assert pv.panel_efficiency == 0.18

    def test_rejects_zero_rated_power(self):
        with pytest.raises(ValidationError):
            PVConfiguration(rated_power_kw=0.0, generation_source="forecast")

    def test_rejects_negative_rated_power(self):
        with pytest.raises(ValidationError):
            PVConfiguration(rated_power_kw=-10.0, generation_source="forecast")

    def test_rejects_negative_max_export(self):
        with pytest.raises(ValidationError):
            PVConfiguration(rated_power_kw=100.0, generation_source="forecast", max_export_kw=-1.0)

    def test_none_max_export_is_valid(self):
        pv = PVConfiguration(rated_power_kw=100.0, generation_source="forecast", max_export_kw=None)
        assert pv.max_export_kw is None

    def test_rejects_invalid_efficiency_zero(self):
        with pytest.raises(ValidationError):
            PVConfiguration(
                rated_power_kw=100.0, generation_source="forecast", panel_efficiency=0.0
            )

    def test_rejects_invalid_efficiency_above_one(self):
        with pytest.raises(ValidationError):
            PVConfiguration(
                rated_power_kw=100.0, generation_source="forecast", panel_efficiency=1.5
            )

    def test_valid_efficiency_boundary(self):
        pv = PVConfiguration(
            rated_power_kw=100.0, generation_source="forecast", panel_efficiency=1.0
        )
        assert pv.panel_efficiency == 1.0


# ---------------------------------------------------------------------------
# BatteryConfiguration
# ---------------------------------------------------------------------------


class TestBatteryConfiguration:
    def test_valid_battery_config(self):
        bat = BatteryConfiguration(rated_power_mw=100.0, energy_capacity_mwh=600.0)
        assert bat.rated_power_mw == 100.0
        assert bat.min_soc_percent == 10.0
        assert bat.charge_efficiency == 0.97

    def test_rejects_zero_power(self):
        with pytest.raises(ValidationError):
            BatteryConfiguration(rated_power_mw=0.0, energy_capacity_mwh=600.0)

    def test_rejects_zero_energy(self):
        with pytest.raises(ValidationError):
            BatteryConfiguration(rated_power_mw=100.0, energy_capacity_mwh=0.0)

    def test_rejects_soc_out_of_range(self):
        with pytest.raises(ValidationError):
            BatteryConfiguration(
                rated_power_mw=100.0, energy_capacity_mwh=600.0, min_soc_percent=-1.0
            )

    def test_rejects_soc_above_100(self):
        with pytest.raises(ValidationError):
            BatteryConfiguration(
                rated_power_mw=100.0, energy_capacity_mwh=600.0, max_soc_percent=101.0
            )

    def test_rejects_efficiency_zero(self):
        with pytest.raises(ValidationError):
            BatteryConfiguration(
                rated_power_mw=100.0, energy_capacity_mwh=600.0, charge_efficiency=0.0
            )

    def test_rejects_efficiency_above_one(self):
        with pytest.raises(ValidationError):
            BatteryConfiguration(
                rated_power_mw=100.0, energy_capacity_mwh=600.0, discharge_efficiency=1.5
            )

    def test_rejects_min_soc_greater_than_max(self):
        with pytest.raises(ValidationError):
            BatteryConfiguration(
                rated_power_mw=100.0,
                energy_capacity_mwh=600.0,
                min_soc_percent=80.0,
                max_soc_percent=20.0,
            )

    def test_rejects_current_energy_out_of_bounds(self):
        with pytest.raises(ValidationError):
            BatteryConfiguration(
                rated_power_mw=100.0,
                energy_capacity_mwh=600.0,
                min_soc_percent=10.0,
                max_soc_percent=90.0,
                current_energy_mwh=1.0,  # Below min_energy = 60 MWh
            )

    def test_valid_current_energy_within_bounds(self):
        bat = BatteryConfiguration(
            rated_power_mw=100.0,
            energy_capacity_mwh=600.0,
            current_energy_mwh=300.0,
        )
        assert bat.current_energy_mwh == 300.0

    def test_none_current_soc_is_valid(self):
        bat = BatteryConfiguration(
            rated_power_mw=100.0, energy_capacity_mwh=600.0, current_soc_percent=None
        )
        assert bat.current_soc_percent is None


# ---------------------------------------------------------------------------
# OptimizationConfiguration
# ---------------------------------------------------------------------------


class TestOptimizationConfiguration:
    def test_valid_optimization_config(self):
        opt = OptimizationConfiguration(optimization_date=date(2026, 1, 1))
        assert opt.duration_hours == 24
        assert opt.time_step_minutes == 15
        assert opt.start_time == "00:00"

    def test_rejects_zero_duration(self):
        with pytest.raises(ValidationError):
            OptimizationConfiguration(optimization_date=date(2026, 1, 1), duration_hours=0)

    def test_rejects_duration_above_48(self):
        with pytest.raises(ValidationError):
            OptimizationConfiguration(optimization_date=date(2026, 1, 1), duration_hours=49)

    def test_rejects_invalid_time_step(self):
        with pytest.raises(ValidationError):
            OptimizationConfiguration(optimization_date=date(2026, 1, 1), time_step_minutes=20)

    def test_valid_time_step_30(self):
        opt = OptimizationConfiguration(optimization_date=date(2026, 1, 1), time_step_minutes=30)
        assert opt.time_step_minutes == 30

    def test_valid_time_step_60(self):
        opt = OptimizationConfiguration(optimization_date=date(2026, 1, 1), time_step_minutes=60)
        assert opt.time_step_minutes == 60

    def test_rejects_invalid_time_format(self):
        with pytest.raises(ValidationError):
            OptimizationConfiguration(optimization_date=date(2026, 1, 1), start_time="25:00")

    def test_rejects_malformed_time(self):
        with pytest.raises(ValidationError):
            OptimizationConfiguration(optimization_date=date(2026, 1, 1), start_time="abc")


# ---------------------------------------------------------------------------
# OutputConfiguration
# ---------------------------------------------------------------------------


class TestOutputConfiguration:
    def test_default_output_config(self):
        out = OutputConfiguration()
        assert out.output_csv_path == "optimization_results.csv"
        assert out.verbose is False
        assert out.currency == "GBP"


# ---------------------------------------------------------------------------
# OptimizationConfig (top-level)
# ---------------------------------------------------------------------------


class TestOptimizationConfig:
    @pytest.fixture()
    def minimal_config(self):
        return OptimizationConfig(
            config_name="test",
            battery=BatteryConfiguration(rated_power_mw=100.0, energy_capacity_mwh=600.0),
            optimization=OptimizationConfiguration(optimization_date=date(2026, 1, 1)),
        )

    def test_creation(self, minimal_config):
        assert minimal_config.config_name == "test"
        assert minimal_config.created_date is not None

    def test_has_pv_false_when_none(self, minimal_config):
        assert minimal_config.has_pv is False

    def test_has_pv_true_when_set(self):
        cfg = OptimizationConfig(
            config_name="test",
            battery=BatteryConfiguration(rated_power_mw=100.0, energy_capacity_mwh=600.0),
            optimization=OptimizationConfiguration(optimization_date=date(2026, 1, 1)),
            pv=PVConfiguration(rated_power_kw=100.0, generation_source="forecast"),
        )
        assert cfg.has_pv is True

    def test_optimization_start_datetime(self, minimal_config):
        dt = minimal_config.optimization_start_datetime
        assert dt == datetime(2026, 1, 1, 0, 0)

    def test_optimization_end_datetime(self, minimal_config):
        dt = minimal_config.optimization_end_datetime
        assert dt == datetime(2026, 1, 2, 0, 0)

    def test_total_time_steps(self, minimal_config):
        assert minimal_config.total_time_steps == 96  # 24h * 60 / 15

    def test_default_created_date_set(self, minimal_config):
        today = datetime.now().strftime("%Y-%m-%d")
        assert minimal_config.created_date == today


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------


class TestConfigIO:
    def test_save_and_load_roundtrip(self, tmp_path):
        cfg = OptimizationConfig(
            config_name="roundtrip-test",
            battery=BatteryConfiguration(rated_power_mw=50.0, energy_capacity_mwh=200.0),
            optimization=OptimizationConfiguration(optimization_date=date(2026, 3, 1)),
        )
        json_path = str(tmp_path / "config.json")
        save_config_to_json(cfg, json_path)

        loaded = load_config_from_json(json_path)
        assert loaded.config_name == "roundtrip-test"
        assert loaded.battery.rated_power_mw == 50.0

    def test_load_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            load_config_from_json("/nonexistent/path.json")

    def test_save_creates_directories(self, tmp_path):
        cfg = OptimizationConfig(
            config_name="nested",
            battery=BatteryConfiguration(rated_power_mw=50.0, energy_capacity_mwh=200.0),
            optimization=OptimizationConfiguration(optimization_date=date(2026, 1, 1)),
        )
        json_path = str(tmp_path / "sub" / "dir" / "config.json")
        save_config_to_json(cfg, json_path)
        assert (tmp_path / "sub" / "dir" / "config.json").exists()
