"""Tests for PV domain models."""

from datetime import datetime

import pytest

from src.ors.domain.models.pv import PVSpec, PVState, PVTelemetry


class TestPVSpec:
    def test_valid_minimal(self):
        spec = PVSpec(rated_power_kw=100.0, max_export_kw=80.0)
        assert spec.rated_power_kw == 100.0
        assert spec.max_export_kw == 80.0

    def test_defaults(self):
        spec = PVSpec(rated_power_kw=100.0, max_export_kw=None)
        assert spec.min_generation_kw == 0.0
        assert spec.curtailment_supported is True

    def test_rejects_negative_rated_power(self):
        with pytest.raises(ValueError, match="rated_power_kw must be positive"):
            PVSpec(rated_power_kw=-1.0, max_export_kw=None)

    def test_rejects_zero_rated_power(self):
        with pytest.raises(ValueError, match="rated_power_kw must be positive"):
            PVSpec(rated_power_kw=0.0, max_export_kw=None)

    def test_rejects_negative_max_export(self):
        with pytest.raises(ValueError, match="max_export_kw must be non-negative"):
            PVSpec(rated_power_kw=100.0, max_export_kw=-1.0)

    def test_rejects_negative_min_generation(self):
        with pytest.raises(ValueError, match="min_generation_kw must be non-negative"):
            PVSpec(rated_power_kw=100.0, max_export_kw=None, min_generation_kw=-5.0)

    def test_rejects_negative_panel_area(self):
        with pytest.raises(ValueError, match="panel_area_m2 must be positive"):
            PVSpec(rated_power_kw=100.0, max_export_kw=None, panel_area_m2=-10.0)

    def test_rejects_zero_panel_area(self):
        with pytest.raises(ValueError, match="panel_area_m2 must be positive"):
            PVSpec(rated_power_kw=100.0, max_export_kw=None, panel_area_m2=0.0)

    def test_rejects_invalid_panel_efficiency(self):
        with pytest.raises(ValueError, match="panel_efficiency"):
            PVSpec(rated_power_kw=100.0, max_export_kw=None, panel_efficiency=1.5)

    def test_rejects_negative_panel_efficiency(self):
        with pytest.raises(ValueError, match="panel_efficiency"):
            PVSpec(rated_power_kw=100.0, max_export_kw=None, panel_efficiency=-0.1)

    def test_rejects_negative_dc_capacity(self):
        with pytest.raises(ValueError, match="dc_capacity_kw must be positive"):
            PVSpec(rated_power_kw=100.0, max_export_kw=None, dc_capacity_kw=-1.0)

    def test_rejects_negative_ac_capacity(self):
        with pytest.raises(ValueError, match="ac_capacity_kw must be positive"):
            PVSpec(rated_power_kw=100.0, max_export_kw=None, ac_capacity_kw=0.0)

    def test_rejects_negative_dc_ac_ratio(self):
        with pytest.raises(ValueError, match="dc_ac_ratio must be positive"):
            PVSpec(rated_power_kw=100.0, max_export_kw=None, dc_ac_ratio=-0.5)

    def test_rejects_invalid_inverter_efficiency(self):
        with pytest.raises(ValueError, match="inverter_efficiency"):
            PVSpec(rated_power_kw=100.0, max_export_kw=None, inverter_efficiency=2.0)

    def test_rejects_invalid_performance_ratio(self):
        with pytest.raises(ValueError, match="performance_ratio"):
            PVSpec(rated_power_kw=100.0, max_export_kw=None, performance_ratio=-0.1)

    def test_rejects_negative_degradation(self):
        with pytest.raises(ValueError, match="degradation_per_year"):
            PVSpec(rated_power_kw=100.0, max_export_kw=None, degradation_per_year=-1.0)

    def test_rejects_invalid_clipping_loss(self):
        with pytest.raises(ValueError, match="clipping_loss_factor"):
            PVSpec(rated_power_kw=100.0, max_export_kw=None, clipping_loss_factor=1.5)

    def test_rejects_invalid_availability(self):
        with pytest.raises(ValueError, match="availability"):
            PVSpec(rated_power_kw=100.0, max_export_kw=None, availability=-0.1)

    def test_rejects_negative_outage_duration(self):
        with pytest.raises(ValueError, match="forced_outage_duration_h"):
            PVSpec(rated_power_kw=100.0, max_export_kw=None, forced_outage_duration_h=-1.0)


class TestPVTelemetry:
    def test_creation(self):
        t = PVTelemetry(timestamp=datetime(2026, 1, 1), generation_kw=50.0)
        assert t.generation_kw == 50.0
        assert t.solar_radiance_kw_per_m2 is None


class TestPVState:
    def test_creation(self):
        state = PVState(
            timestamp=datetime(2026, 1, 1),
            generation_kw=100.0,
            energy_kwh=25.0,
            curtailed_kw=0.0,
            curtailed=False,
            exportable_kw=100.0,
            exportable_kwh=25.0,
            solar_radiance_kw_per_m2=None,
            estimated_from_radiance=False,
        )
        assert state.generation_kw == 100.0
        assert state.quality_flags == set()
