"""Tests for battery state integration with optimizer.

Tests the new functionality that allows the optimizer to accept
real-time battery state instead of hardcoded 50% SOC.
"""

from datetime import datetime

import pytest
from src.ors.domain.models.battery import BatterySpec, BatteryTelemetry

battery_status = pytest.importorskip(
    "src.ors.services.battery_status",
    reason="battery_status module not yet implemented",
)
update_battery_state = battery_status.update_battery_state
from src.ors.services.optimizer.optimizer import extract_optimizer_initial_state


class TestBatteryStateIntegration:
    """Test integration of real battery state with optimizer."""

    @pytest.fixture
    def battery_spec(self) -> BatterySpec:
        """Create test battery specification."""
        return BatterySpec(
            rated_power_mw=100.0,
            energy_capacity_mwh=600.0,
            min_soc_percent=10.0,
            max_soc_percent=90.0,
            charge_efficiency=0.97,
            discharge_efficiency=0.97,
            auxiliary_power_mw=0.5,
            self_discharge_rate_per_hour=0.0005,
        )

    def test_battery_telemetry_validation(self, battery_spec):
        """Test that battery telemetry is properly validated."""
        telemetry = BatteryTelemetry(
            timestamp=datetime(2026, 3, 7, 8, 0, 0),
            current_energy_mwh=300.0,
            current_soc_percent=50.0,
            current_power_mw=0.0,
            operating_mode="idle",
            is_available=True,
        )

        # Process telemetry
        state = update_battery_state(battery_spec, telemetry)

        # Verify correct processing
        assert state.energy_mwh == 300.0
        assert state.soc_percent == 50.0
        assert state.operating_mode == "idle"
        assert state.is_available is True
        assert len(state.quality_flags) == 0  # No quality issues

    def test_missing_telemetry_fallback(self, battery_spec):
        """Test fallback behavior when telemetry is missing."""
        telemetry = BatteryTelemetry(
            timestamp=datetime(2026, 3, 7, 8, 0, 0),
            current_energy_mwh=None,  # Missing
            current_soc_percent=None,  # Missing
            current_power_mw=None,  # Missing
            operating_mode=None,  # Missing
            is_available=True,
        )

        state = update_battery_state(battery_spec, telemetry)

        # Should fall back to 50% SOC
        assert state.soc_percent == 50.0
        assert state.energy_mwh == 300.0  # 50% of 600 MWh
        assert state.operating_mode == "idle"

        # Should have quality flags for missing data
        assert "missing_energy_and_soc" in state.quality_flags
        assert "defaulted_to_50_percent_soc" in state.quality_flags
        assert "missing_power_data" in state.quality_flags
        assert "mode_estimated_from_power" in state.quality_flags

    def test_extract_optimizer_state_with_none(self):
        """Test extracting optimizer state when no battery state provided."""
        energy, mode, cycles = extract_optimizer_initial_state(None)

        # Should use defaults
        assert energy == 300.0  # 50% of 600 MWh (E_CAP * 0.5)
        assert mode == "idle"
        assert cycles == 0

    def test_energy_soc_consistency(self, battery_spec):
        """Test that energy and SOC calculations are consistent."""
        # Test various SOC levels
        test_cases = [
            (10.0, 60.0),  # 10% SOC = 60 MWh
            (50.0, 300.0),  # 50% SOC = 300 MWh
            (90.0, 540.0),  # 90% SOC = 540 MWh
        ]

        for soc_percent, expected_energy in test_cases:
            telemetry = BatteryTelemetry(
                timestamp=datetime(2026, 3, 7, 8, 0, 0),
                current_soc_percent=soc_percent,
                current_power_mw=0.0,
                operating_mode="idle",
                is_available=True,
            )

            state = update_battery_state(battery_spec, telemetry)

            assert state.soc_percent == soc_percent
            assert abs(state.energy_mwh - expected_energy) < 0.1  # Within 0.1 MWh

    def test_operating_mode_detection(self, battery_spec):
        """Test that operating mode is correctly determined from power."""
        test_cases = [
            (50.0, "charging"),  # Positive power = charging
            (-30.0, "discharging"),  # Negative power = discharging
            (0.05, "idle"),  # Small power = idle
            (-0.05, "idle"),  # Small negative power = idle
        ]

        for power_mw, expected_mode in test_cases:
            telemetry = BatteryTelemetry(
                timestamp=datetime(2026, 3, 7, 8, 0, 0),
                current_energy_mwh=300.0,
                current_power_mw=power_mw,
                operating_mode=None,  # Let it be estimated
                is_available=True,
            )

            state = update_battery_state(battery_spec, telemetry)
            assert state.operating_mode == expected_mode

    def test_soc_energy_bounds_clamping(self, battery_spec):
        """Test that energy is properly clamped within SOC bounds."""
        # Test below minimum (10% = 60 MWh)
        telemetry_low = BatteryTelemetry(
            timestamp=datetime(2026, 3, 7, 8, 0, 0),
            current_energy_mwh=30.0,  # Below 10% minimum
            operating_mode="idle",
            is_available=True,
        )

        state_low = update_battery_state(battery_spec, telemetry_low)
        assert state_low.energy_mwh == 60.0  # Clamped to minimum
        assert state_low.soc_percent == 10.0
        assert "energy_clamped_to_minimum" in state_low.quality_flags

        # Test above maximum (90% = 540 MWh)
        telemetry_high = BatteryTelemetry(
            timestamp=datetime(2026, 3, 7, 8, 0, 0),
            current_energy_mwh=580.0,  # Above 90% maximum
            operating_mode="idle",
            is_available=True,
        )

        state_high = update_battery_state(battery_spec, telemetry_high)
        assert state_high.energy_mwh == 540.0  # Clamped to maximum
        assert state_high.soc_percent == 90.0
        assert "energy_clamped_to_maximum" in state_high.quality_flags


class TestOptimizerBatteryStateIntegration:
    """Test that optimizer properly accepts battery state inputs."""

    def test_build_model_accepts_initial_state(self):
        """Test that build_model accepts initial battery state parameters."""
        from src.ors.services.optimizer.optimizer import build_model

        # Create simple test data
        price = dict.fromkeys(range(1, 5), 50.0)  # 4 timesteps
        solar = dict.fromkeys(range(1, 5), 10.0)
        p_30 = 60.0

        # Test with custom initial state
        model = build_model(
            price=price,
            solar=solar,
            p_30=p_30,
            initial_energy_mwh=180.0,  # 30% SOC
            initial_mode="charging",
            cycles_used_today=1,
        )

        # Verify model was created successfully
        assert model is not None
        assert len(list(model.T)) == 4  # 4 timesteps

        # Note: More detailed testing would require solving the model
        # and checking that initial energy constraint is properly set
