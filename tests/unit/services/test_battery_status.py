"""Tests for battery_status module."""

from datetime import datetime

import pytest
from src.ors.domain.models.battery import BatterySpec, BatteryState, BatteryTelemetry
from src.ors.services.battery.battery_status import (
    determine_operating_mode,
    estimate_energy_from_soc,
    estimate_soc_from_energy,
    update_battery_state,
)


@pytest.fixture()
def spec():
    return BatterySpec(rated_power_mw=100.0, energy_capacity_mwh=600.0)


# ---------------------------------------------------------------------------
# estimate_energy_from_soc
# ---------------------------------------------------------------------------


class TestEstimateEnergyFromSoc:
    def test_zero_soc(self):
        assert estimate_energy_from_soc(0.0, 600.0) == pytest.approx(0.0)

    def test_full_soc(self):
        assert estimate_energy_from_soc(100.0, 600.0) == pytest.approx(600.0)

    def test_half_soc(self):
        assert estimate_energy_from_soc(50.0, 600.0) == pytest.approx(300.0)

    def test_invalid_soc_negative(self):
        with pytest.raises(ValueError, match="between 0 and 100"):
            estimate_energy_from_soc(-1.0, 600.0)

    def test_invalid_soc_over_100(self):
        with pytest.raises(ValueError, match="between 0 and 100"):
            estimate_energy_from_soc(101.0, 600.0)

    def test_invalid_capacity_zero(self):
        with pytest.raises(ValueError, match="positive"):
            estimate_energy_from_soc(50.0, 0.0)

    def test_invalid_capacity_negative(self):
        with pytest.raises(ValueError, match="positive"):
            estimate_energy_from_soc(50.0, -100.0)


# ---------------------------------------------------------------------------
# estimate_soc_from_energy
# ---------------------------------------------------------------------------


class TestEstimateSocFromEnergy:
    def test_zero_energy(self):
        assert estimate_soc_from_energy(0.0, 600.0) == pytest.approx(0.0)

    def test_full_energy(self):
        assert estimate_soc_from_energy(600.0, 600.0) == pytest.approx(100.0)

    def test_invalid_negative_energy(self):
        with pytest.raises(ValueError, match="non-negative"):
            estimate_soc_from_energy(-1.0, 600.0)

    def test_invalid_zero_capacity(self):
        with pytest.raises(ValueError, match="positive"):
            estimate_soc_from_energy(100.0, 0.0)


# ---------------------------------------------------------------------------
# determine_operating_mode
# ---------------------------------------------------------------------------


class TestDetermineOperatingMode:
    def test_idle_zero_power(self):
        assert determine_operating_mode(0.0) == "idle"

    def test_idle_below_threshold(self):
        assert determine_operating_mode(0.05) == "idle"

    def test_charging(self):
        assert determine_operating_mode(50.0) == "charging"

    def test_discharging(self):
        assert determine_operating_mode(-50.0) == "discharging"

    def test_custom_threshold(self):
        assert determine_operating_mode(0.5, power_threshold_mw=1.0) == "idle"
        assert determine_operating_mode(1.5, power_threshold_mw=1.0) == "charging"


# ---------------------------------------------------------------------------
# update_battery_state
# ---------------------------------------------------------------------------


class TestUpdateBatteryState:
    def test_basic_with_energy_telemetry(self, spec):
        telemetry = BatteryTelemetry(
            timestamp=datetime(2026, 1, 1),
            current_energy_mwh=300.0,
            current_power_mw=0.0,
            operating_mode="idle",
        )
        state = update_battery_state(spec, telemetry)
        assert state.energy_mwh == pytest.approx(300.0)
        assert state.operating_mode == "idle"

    def test_soc_only_telemetry(self, spec):
        telemetry = BatteryTelemetry(
            timestamp=datetime(2026, 1, 1),
            current_soc_percent=50.0,
        )
        state = update_battery_state(spec, telemetry)
        assert state.energy_mwh == pytest.approx(300.0)
        assert "energy_estimated_from_soc" in state.quality_flags

    def test_missing_both_energy_and_soc_defaults(self, spec):
        telemetry = BatteryTelemetry(timestamp=datetime(2026, 1, 1))
        state = update_battery_state(spec, telemetry)
        assert "defaulted_to_50_percent_soc" in state.quality_flags
        assert state.soc_percent == pytest.approx(50.0)

    def test_missing_both_with_prev_state(self, spec):
        prev = BatteryState(
            timestamp=datetime(2026, 1, 1, 0, 0),
            energy_mwh=200.0,
            soc_percent=33.33,
            power_mw=0.0,
            operating_mode="idle",
            is_available=True,
            estimated_values=set(),
            quality_flags=set(),
        )
        telemetry = BatteryTelemetry(timestamp=datetime(2026, 1, 1, 1, 0))
        state = update_battery_state(spec, telemetry, prev_state=prev)
        assert "using_previous_state" in state.quality_flags
        assert state.energy_mwh == pytest.approx(200.0)

    def test_energy_clamped_to_min(self, spec):
        telemetry = BatteryTelemetry(
            timestamp=datetime(2026, 1, 1),
            current_energy_mwh=10.0,  # Below min_energy = 60 MWh
        )
        state = update_battery_state(spec, telemetry)
        assert "energy_clamped_to_minimum" in state.quality_flags
        assert state.energy_mwh == pytest.approx(spec.min_energy_mwh)

    def test_energy_clamped_to_max(self, spec):
        telemetry = BatteryTelemetry(
            timestamp=datetime(2026, 1, 1),
            current_energy_mwh=560.0,  # Above max_energy = 540 MWh
        )
        state = update_battery_state(spec, telemetry)
        assert "energy_clamped_to_maximum" in state.quality_flags
        assert state.energy_mwh == pytest.approx(spec.max_energy_mwh)

    def test_power_clamped_to_rated(self, spec):
        telemetry = BatteryTelemetry(
            timestamp=datetime(2026, 1, 1),
            current_energy_mwh=300.0,
            current_power_mw=200.0,  # Above rated 100 MW
        )
        state = update_battery_state(spec, telemetry)
        assert "power_clamped_to_rated" in state.quality_flags
        assert state.power_mw == pytest.approx(100.0)

    def test_missing_power_data(self, spec):
        telemetry = BatteryTelemetry(
            timestamp=datetime(2026, 1, 1),
            current_energy_mwh=300.0,
        )
        state = update_battery_state(spec, telemetry)
        assert "missing_power_data" in state.quality_flags

    def test_mode_power_mismatch(self, spec):
        telemetry = BatteryTelemetry(
            timestamp=datetime(2026, 1, 1),
            current_energy_mwh=300.0,
            current_power_mw=-50.0,  # Negative = discharging
            operating_mode="charging",  # Mismatch
        )
        state = update_battery_state(spec, telemetry)
        assert "mode_power_mismatch" in state.quality_flags
        assert state.operating_mode == "discharging"  # Power wins

    def test_mode_estimated_from_power(self, spec):
        telemetry = BatteryTelemetry(
            timestamp=datetime(2026, 1, 1),
            current_energy_mwh=300.0,
            current_power_mw=50.0,
        )
        state = update_battery_state(spec, telemetry)
        assert "mode_estimated_from_power" in state.quality_flags
        assert state.operating_mode == "charging"

    def test_energy_soc_mismatch_flag(self, spec):
        telemetry = BatteryTelemetry(
            timestamp=datetime(2026, 1, 1),
            current_energy_mwh=300.0,  # = 50% SOC
            current_soc_percent=80.0,  # Mismatch
        )
        state = update_battery_state(spec, telemetry)
        assert "energy_soc_mismatch" in state.quality_flags

    def test_unrealistic_energy_change_flag(self, spec):
        prev = BatteryState(
            timestamp=datetime(2026, 1, 1, 0, 0),
            energy_mwh=300.0,
            soc_percent=50.0,
            power_mw=0.0,
            operating_mode="idle",
            is_available=True,
            estimated_values=set(),
            quality_flags=set(),
        )
        telemetry = BatteryTelemetry(
            timestamp=datetime(2026, 1, 1, 0, 15),  # 15 min later
            current_energy_mwh=500.0,  # 200 MWh change in 15 min, max possible = 25 MWh
        )
        state = update_battery_state(spec, telemetry, prev_state=prev)
        assert "unrealistic_energy_change" in state.quality_flags
