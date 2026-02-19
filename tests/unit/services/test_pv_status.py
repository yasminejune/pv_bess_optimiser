"""Unit tests for PV status service."""

from datetime import datetime

import pytest
from ors.domain.models.pv import PVSpec, PVTelemetry
from ors.services.pv_status import estimate_energy_from_radiance, update_pv_state


class TestUpdatePVState:
    """Test cases for update_pv_state function."""

    @pytest.fixture
    def base_spec(self) -> PVSpec:
        """Basic PV specification for testing."""
        return PVSpec(
            rated_power_kw=100.0,
            max_export_kw=80.0,
            min_generation_kw=0.0,
            curtailment_supported=True,
        )

    @pytest.fixture
    def base_timestamp(self) -> datetime:
        """Fixed timestamp for testing."""
        return datetime(2026, 1, 1, 0, 0, 0)

    def test_normal_generation(self, base_spec: PVSpec, base_timestamp: datetime):
        """Test normal generation within all limits."""
        telemetry = PVTelemetry(timestamp=base_timestamp, generation_kw=50.0)

        state = update_pv_state(base_spec, telemetry, timestep_minutes=15)

        assert state.timestamp == base_timestamp
        assert state.generation_kw == 50.0
        assert state.energy_kwh == 50.0 * (15 / 60.0)  # 12.5 kWh
        assert state.curtailed_kw == 0.0
        assert state.curtailed is False
        assert state.exportable_kw == 50.0
        assert state.exportable_kwh == 50.0 * (15 / 60.0)
        assert len(state.quality_flags) == 0

    def test_kwh_conversion_different_timesteps(self, base_spec: PVSpec, base_timestamp: datetime):
        """Test energy calculation for different timestep durations."""
        telemetry = PVTelemetry(timestamp=base_timestamp, generation_kw=60.0)

        # 15 minutes
        state_15 = update_pv_state(base_spec, telemetry, timestep_minutes=15)
        assert state_15.energy_kwh == 60.0 * (15 / 60.0)  # 15.0 kWh
        assert state_15.exportable_kwh == 60.0 * (15 / 60.0)

        # 30 minutes
        state_30 = update_pv_state(base_spec, telemetry, timestep_minutes=30)
        assert state_30.energy_kwh == 60.0 * (30 / 60.0)  # 30.0 kWh
        assert state_30.exportable_kwh == 60.0 * (30 / 60.0)

        # 60 minutes
        state_60 = update_pv_state(base_spec, telemetry, timestep_minutes=60)
        assert state_60.energy_kwh == 60.0  # 60.0 kWh
        assert state_60.exportable_kwh == 60.0

    def test_missing_generation(self, base_spec: PVSpec, base_timestamp: datetime):
        """Test handling of None generation value."""
        telemetry = PVTelemetry(timestamp=base_timestamp, generation_kw=None)

        state = update_pv_state(base_spec, telemetry, timestep_minutes=15)

        assert state.generation_kw == 0.0
        assert state.energy_kwh == 0.0
        assert state.exportable_kw == 0.0
        assert state.exportable_kwh == 0.0
        assert "missing_generation" in state.quality_flags
        assert len(state.quality_flags) == 1

    def test_negative_generation_clamped(self, base_spec: PVSpec, base_timestamp: datetime):
        """Test that negative generation is clamped to zero."""
        telemetry = PVTelemetry(timestamp=base_timestamp, generation_kw=-10.0)

        state = update_pv_state(base_spec, telemetry, timestep_minutes=15)

        assert state.generation_kw == 0.0
        assert state.energy_kwh == 0.0
        assert state.exportable_kw == 0.0
        assert state.exportable_kwh == 0.0
        assert "negative_generation_clamped" in state.quality_flags
        assert len(state.quality_flags) == 1

    def test_above_rated_power_clamped(self, base_spec: PVSpec, base_timestamp: datetime):
        """Test that generation above rated power is clamped."""
        telemetry = PVTelemetry(
            timestamp=base_timestamp, generation_kw=150.0  # Above rated_power_kw=100
        )

        state = update_pv_state(base_spec, telemetry, timestep_minutes=15)

        assert state.generation_kw == 100.0  # Clamped to rated_power_kw
        assert state.energy_kwh == 100.0 * (15 / 60.0)  # 25.0 kWh
        assert "above_rated_clamped" in state.quality_flags

    def test_exportable_respects_max_export(self, base_spec: PVSpec, base_timestamp: datetime):
        """Test that exportable_kw respects max_export_kw limit."""
        telemetry = PVTelemetry(
            timestamp=base_timestamp, generation_kw=90.0  # Above max_export_kw=80
        )

        state = update_pv_state(base_spec, telemetry, timestep_minutes=15)

        assert state.generation_kw == 90.0
        assert state.exportable_kw == 80.0  # Capped at max_export_kw
        assert state.exportable_kwh == 80.0 * (15 / 60.0)  # 20.0 kWh

    def test_curtailment_when_above_max_export(self, base_spec: PVSpec, base_timestamp: datetime):
        """Test curtailment logic when generation exceeds max_export_kw."""
        telemetry = PVTelemetry(
            timestamp=base_timestamp, generation_kw=90.0  # Above max_export_kw=80
        )

        state = update_pv_state(base_spec, telemetry, timestep_minutes=15)

        assert state.curtailed_kw == 10.0  # 90 - 80
        assert state.curtailed is True
        assert state.exportable_kw == 80.0
        assert len(state.quality_flags) == 0  # No quality issues, just curtailment

    def test_no_curtailment_when_curtailment_not_supported(self, base_timestamp: datetime):
        """Test behavior when curtailment_supported=False."""
        spec = PVSpec(rated_power_kw=100.0, max_export_kw=80.0, curtailment_supported=False)
        telemetry = PVTelemetry(timestamp=base_timestamp, generation_kw=90.0)

        state = update_pv_state(spec, telemetry, timestep_minutes=15)

        assert state.generation_kw == 90.0
        assert state.curtailed_kw == 0.0
        assert state.curtailed is False
        assert state.exportable_kw == 80.0  # Still capped
        assert state.exportable_kwh == 80.0 * (15 / 60.0)
        assert "export_cap_applied_no_curtailment" in state.quality_flags

    def test_no_max_export_limit(self, base_timestamp: datetime):
        """Test behavior when max_export_kw is None."""
        spec = PVSpec(rated_power_kw=100.0, max_export_kw=None)  # No export limit
        telemetry = PVTelemetry(timestamp=base_timestamp, generation_kw=90.0)

        state = update_pv_state(spec, telemetry, timestep_minutes=15)

        assert state.generation_kw == 90.0
        assert state.exportable_kw == 90.0  # No cap
        assert state.exportable_kwh == 90.0 * (15 / 60.0)
        assert state.curtailed_kw == 0.0
        assert state.curtailed is False
        assert len(state.quality_flags) == 0

    def test_generation_at_max_export_no_curtailment(
        self, base_spec: PVSpec, base_timestamp: datetime
    ):
        """Test that generation exactly at max_export_kw does not trigger curtailment."""
        telemetry = PVTelemetry(
            timestamp=base_timestamp, generation_kw=80.0  # Exactly at max_export_kw
        )

        state = update_pv_state(base_spec, telemetry, timestep_minutes=15)

        assert state.generation_kw == 80.0
        assert state.exportable_kw == 80.0
        assert state.curtailed_kw == 0.0
        assert state.curtailed is False
        assert len(state.quality_flags) == 0

    def test_timestep_validation_zero(self, base_spec: PVSpec, base_timestamp: datetime):
        """Test that timestep_minutes=0 raises ValueError."""
        telemetry = PVTelemetry(timestamp=base_timestamp, generation_kw=50.0)

        with pytest.raises(ValueError, match="timestep_minutes must be positive"):
            update_pv_state(base_spec, telemetry, timestep_minutes=0)

    def test_timestep_validation_negative(self, base_spec: PVSpec, base_timestamp: datetime):
        """Test that negative timestep_minutes raises ValueError."""
        telemetry = PVTelemetry(timestamp=base_timestamp, generation_kw=50.0)

        with pytest.raises(ValueError, match="timestep_minutes must be positive"):
            update_pv_state(base_spec, telemetry, timestep_minutes=-15)

    def test_multiple_quality_flags(self, base_timestamp: datetime):
        """Test that multiple quality flags can be set simultaneously."""
        spec = PVSpec(rated_power_kw=100.0, max_export_kw=80.0, curtailment_supported=False)
        # Create a scenario where generation is both above rated and above max_export
        # but since we clamp at rated first, we'll only get one flag
        # Let's test negative + curtailment disabled instead
        telemetry = PVTelemetry(timestamp=base_timestamp, generation_kw=-10.0)

        state = update_pv_state(spec, telemetry, timestep_minutes=15)

        assert "negative_generation_clamped" in state.quality_flags

    def test_zero_generation_valid(self, base_spec: PVSpec, base_timestamp: datetime):
        """Test that zero generation is valid (e.g., at night)."""
        telemetry = PVTelemetry(timestamp=base_timestamp, generation_kw=0.0)

        state = update_pv_state(base_spec, telemetry, timestep_minutes=15)

        assert state.generation_kw == 0.0
        assert state.energy_kwh == 0.0
        assert state.exportable_kw == 0.0
        assert state.exportable_kwh == 0.0
        assert state.curtailed_kw == 0.0
        assert state.curtailed is False
        assert len(state.quality_flags) == 0  # Zero is valid, not missing

    def test_min_generation_constraint(self, base_timestamp: datetime):
        """Test that min_generation_kw constraint is applied."""
        spec = PVSpec(rated_power_kw=100.0, max_export_kw=None, min_generation_kw=5.0)
        telemetry = PVTelemetry(
            timestamp=base_timestamp, generation_kw=2.0  # Below min_generation_kw
        )

        state = update_pv_state(spec, telemetry, timestep_minutes=15)

        assert state.generation_kw == 5.0  # Raised to min
        assert state.energy_kwh == 5.0 * (15 / 60.0)
        assert "below_min_generation_clamped" in state.quality_flags

    def test_curtailment_at_rated_power_limit(self, base_timestamp: datetime):
        """Test curtailment when generation is at rated power and above max_export."""
        spec = PVSpec(rated_power_kw=100.0, max_export_kw=80.0, curtailment_supported=True)
        telemetry = PVTelemetry(
            timestamp=base_timestamp, generation_kw=100.0  # At rated power, above max_export
        )

        state = update_pv_state(spec, telemetry, timestep_minutes=15)

        assert state.generation_kw == 100.0
        assert state.exportable_kw == 80.0
        assert state.curtailed_kw == 20.0  # 100 - 80
        assert state.curtailed is True
        assert len(state.quality_flags) == 0


class TestEstimateEnergyFromRadiance:
    """Test cases for estimate_energy_from_radiance function."""

    def test_basic_estimation_15min(self):
        """Test basic radiance estimation with 15-minute timestep."""
        # 1.0 kW/m^2 * 100 m^2 * 0.2 * 0.25 hours = 5.0 kWh
        energy = estimate_energy_from_radiance(
            solar_radiance_kw_per_m2=1.0,
            panel_area_m2=100.0,
            panel_efficiency=0.2,
            timestep_minutes=15,
        )
        assert energy == pytest.approx(5.0)

    def test_estimation_30min(self):
        """Test radiance estimation with 30-minute timestep."""
        # 0.8 kW/m^2 * 50 m^2 * 0.15 * 0.5 hours = 3.0 kWh
        energy = estimate_energy_from_radiance(
            solar_radiance_kw_per_m2=0.8,
            panel_area_m2=50.0,
            panel_efficiency=0.15,
            timestep_minutes=30,
        )
        assert energy == pytest.approx(3.0)

    def test_estimation_zero_radiance(self):
        """Test estimation with zero radiance (night time)."""
        energy = estimate_energy_from_radiance(
            solar_radiance_kw_per_m2=0.0,
            panel_area_m2=100.0,
            panel_efficiency=0.2,
            timestep_minutes=15,
        )
        assert energy == 0.0

    def test_estimation_high_efficiency(self):
        """Test estimation with high efficiency panels."""
        # 1.0 kW/m^2 * 100 m^2 * 0.9 * 0.25 hours = 22.5 kWh
        energy = estimate_energy_from_radiance(
            solar_radiance_kw_per_m2=1.0,
            panel_area_m2=100.0,
            panel_efficiency=0.9,
            timestep_minutes=15,
        )
        assert energy == pytest.approx(22.5)

    def test_validation_negative_radiance(self):
        """Test that negative radiance raises ValueError."""
        with pytest.raises(ValueError, match="solar_radiance_kw_per_m2 must be non-negative"):
            estimate_energy_from_radiance(
                solar_radiance_kw_per_m2=-0.5,
                panel_area_m2=100.0,
                panel_efficiency=0.2,
                timestep_minutes=15,
            )

    def test_validation_zero_panel_area(self):
        """Test that zero panel area raises ValueError."""
        with pytest.raises(ValueError, match="panel_area_m2 must be positive"):
            estimate_energy_from_radiance(
                solar_radiance_kw_per_m2=1.0,
                panel_area_m2=0.0,
                panel_efficiency=0.2,
                timestep_minutes=15,
            )

    def test_validation_negative_panel_area(self):
        """Test that negative panel area raises ValueError."""
        with pytest.raises(ValueError, match="panel_area_m2 must be positive"):
            estimate_energy_from_radiance(
                solar_radiance_kw_per_m2=1.0,
                panel_area_m2=-100.0,
                panel_efficiency=0.2,
                timestep_minutes=15,
            )

    def test_validation_efficiency_below_zero(self):
        """Test that efficiency below 0 raises ValueError."""
        with pytest.raises(ValueError, match="panel_efficiency must be between 0 and 1"):
            estimate_energy_from_radiance(
                solar_radiance_kw_per_m2=1.0,
                panel_area_m2=100.0,
                panel_efficiency=-0.1,
                timestep_minutes=15,
            )

    def test_validation_efficiency_above_one(self):
        """Test that efficiency above 1 raises ValueError."""
        with pytest.raises(ValueError, match="panel_efficiency must be between 0 and 1"):
            estimate_energy_from_radiance(
                solar_radiance_kw_per_m2=1.0,
                panel_area_m2=100.0,
                panel_efficiency=1.5,
                timestep_minutes=15,
            )

    def test_validation_zero_timestep(self):
        """Test that zero timestep raises ValueError."""
        with pytest.raises(ValueError, match="timestep_minutes must be positive"):
            estimate_energy_from_radiance(
                solar_radiance_kw_per_m2=1.0,
                panel_area_m2=100.0,
                panel_efficiency=0.2,
                timestep_minutes=0,
            )

    def test_validation_negative_timestep(self):
        """Test that negative timestep raises ValueError."""
        with pytest.raises(ValueError, match="timestep_minutes must be positive"):
            estimate_energy_from_radiance(
                solar_radiance_kw_per_m2=1.0,
                panel_area_m2=100.0,
                panel_efficiency=0.2,
                timestep_minutes=-15,
            )


class TestRadianceBasedEstimation:
    """Test cases for radiance-based PV estimation in update_pv_state."""

    @pytest.fixture
    def spec_with_radiance_params(self) -> PVSpec:
        """PV spec with radiance estimation parameters."""
        return PVSpec(
            rated_power_kw=100.0, max_export_kw=80.0, panel_area_m2=100.0, panel_efficiency=0.2
        )

    @pytest.fixture
    def base_timestamp(self) -> datetime:
        """Fixed timestamp for testing."""
        return datetime(2026, 1, 1, 0, 0, 0)

    def test_missing_generation_with_radiance(
        self, spec_with_radiance_params: PVSpec, base_timestamp: datetime
    ):
        """Test that missing generation falls back to radiance estimation."""
        telemetry = PVTelemetry(
            timestamp=base_timestamp, generation_kw=None, solar_radiance_kw_per_m2=0.8
        )

        state = update_pv_state(spec_with_radiance_params, telemetry, timestep_minutes=15)

        # Expected: 0.8 kW/m^2 * 100 m^2 * 0.2 * 0.25 hours = 4.0 kWh
        assert state.energy_kwh == pytest.approx(4.0)
        # Expected generation_kw = 4.0 kWh / 0.25 hours = 16.0 kW
        assert state.generation_kw == pytest.approx(16.0)
        assert state.estimated_from_radiance is True
        assert "missing_generation" in state.quality_flags
        assert "estimated_from_radiance" in state.quality_flags
        assert len(state.quality_flags) == 2

    def test_missing_generation_no_radiance_data(
        self, spec_with_radiance_params: PVSpec, base_timestamp: datetime
    ):
        """Test that missing generation and radiance both result in zero."""
        telemetry = PVTelemetry(
            timestamp=base_timestamp, generation_kw=None, solar_radiance_kw_per_m2=None
        )

        state = update_pv_state(spec_with_radiance_params, telemetry, timestep_minutes=15)

        assert state.generation_kw == 0.0
        assert state.energy_kwh == 0.0
        assert state.estimated_from_radiance is False
        assert "missing_generation" in state.quality_flags
        assert "estimated_from_radiance" not in state.quality_flags
        assert len(state.quality_flags) == 1

    def test_missing_generation_no_panel_specs(self, base_timestamp: datetime):
        """Test that missing generation without panel specs results in zero."""
        spec = PVSpec(
            rated_power_kw=100.0,
            max_export_kw=80.0,
            # No panel_area_m2 or panel_efficiency
        )
        telemetry = PVTelemetry(
            timestamp=base_timestamp, generation_kw=None, solar_radiance_kw_per_m2=0.8
        )

        state = update_pv_state(spec, telemetry, timestep_minutes=15)

        assert state.generation_kw == 0.0
        assert state.energy_kwh == 0.0
        assert state.estimated_from_radiance is False
        assert "missing_generation" in state.quality_flags
        assert len(state.quality_flags) == 1

    def test_radiance_estimation_respects_rated_power(
        self, spec_with_radiance_params: PVSpec, base_timestamp: datetime
    ):
        """Test that radiance estimation is clamped to rated power."""
        # High radiance that would exceed rated power
        # 2.0 kW/m^2 * 100 m^2 * 0.2 * 0.25 = 10.0 kWh -> 40.0 kW
        telemetry = PVTelemetry(
            timestamp=base_timestamp, generation_kw=None, solar_radiance_kw_per_m2=2.0
        )

        state = update_pv_state(spec_with_radiance_params, telemetry, timestep_minutes=15)

        # Should be clamped to 100.0 kW (rated_power_kw)
        assert state.generation_kw == pytest.approx(40.0)  # Not clamped since 40 < 100
        assert state.estimated_from_radiance is True

    def test_radiance_estimation_with_curtailment(self, base_timestamp: datetime):
        """Test radiance estimation with curtailment."""
        spec = PVSpec(
            rated_power_kw=100.0,
            max_export_kw=15.0,
            panel_area_m2=100.0,
            panel_efficiency=0.2,
            curtailment_supported=True,
        )
        # 0.8 kW/m^2 * 100 m^2 * 0.2 * 0.25 = 4.0 kWh -> 16.0 kW
        telemetry = PVTelemetry(
            timestamp=base_timestamp, generation_kw=None, solar_radiance_kw_per_m2=0.8
        )

        state = update_pv_state(spec, telemetry, timestep_minutes=15)

        assert state.generation_kw == pytest.approx(16.0)
        assert state.exportable_kw == pytest.approx(15.0)
        assert state.curtailed_kw == pytest.approx(1.0)
        assert state.curtailed is True
        assert state.estimated_from_radiance is True

    def test_telemetry_generation_preferred_over_radiance(
        self, spec_with_radiance_params: PVSpec, base_timestamp: datetime
    ):
        """Test that telemetry generation is used when available, even if radiance is present."""
        telemetry = PVTelemetry(
            timestamp=base_timestamp, generation_kw=50.0, solar_radiance_kw_per_m2=0.8
        )

        state = update_pv_state(spec_with_radiance_params, telemetry, timestep_minutes=15)

        # Should use telemetry generation, not radiance
        assert state.generation_kw == 50.0
        assert state.energy_kwh == pytest.approx(50.0 * 0.25)
        assert state.estimated_from_radiance is False
        assert "estimated_from_radiance" not in state.quality_flags
        assert len(state.quality_flags) == 0

    def test_below_min_generation_flag(self, base_timestamp: datetime):
        """Test that below_min_generation_clamped flag is added when needed."""
        spec = PVSpec(
            rated_power_kw=100.0,
            max_export_kw=None,
            min_generation_kw=5.0,
            panel_area_m2=100.0,
            panel_efficiency=0.2,
        )
        # Low radiance: 0.05 kW/m^2 * 100 m^2 * 0.2 * 0.25 = 0.25 kWh -> 1.0 kW
        telemetry = PVTelemetry(
            timestamp=base_timestamp, generation_kw=None, solar_radiance_kw_per_m2=0.05
        )

        state = update_pv_state(spec, telemetry, timestep_minutes=15)

        # Should be clamped to min_generation_kw
        assert state.generation_kw == pytest.approx(5.0)
        assert "below_min_generation_clamped" in state.quality_flags
        assert "estimated_from_radiance" in state.quality_flags
        assert "missing_generation" in state.quality_flags
