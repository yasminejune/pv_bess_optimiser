"""Tests for battery domain models."""

from datetime import datetime

import pytest
from pydantic import ValidationError
from src.ors.domain.models.battery import BatterySpec, BatteryState, BatteryTelemetry


class TestBatterySpec:
    def test_valid_creation(self):
        spec = BatterySpec(rated_power_mw=100.0, energy_capacity_mwh=600.0)
        assert spec.rated_power_mw == 100.0
        assert spec.charge_efficiency == 0.97

    def test_rejects_zero_power(self):
        with pytest.raises(ValidationError):
            BatterySpec(rated_power_mw=0.0, energy_capacity_mwh=600.0)

    def test_rejects_negative_energy(self):
        with pytest.raises(ValidationError):
            BatterySpec(rated_power_mw=100.0, energy_capacity_mwh=-1.0)

    def test_rejects_soc_below_zero(self):
        with pytest.raises(ValidationError):
            BatterySpec(rated_power_mw=100.0, energy_capacity_mwh=600.0, min_soc_percent=-1.0)

    def test_rejects_soc_above_100(self):
        with pytest.raises(ValidationError):
            BatterySpec(rated_power_mw=100.0, energy_capacity_mwh=600.0, max_soc_percent=101.0)

    def test_rejects_efficiency_zero(self):
        with pytest.raises(ValidationError):
            BatterySpec(rated_power_mw=100.0, energy_capacity_mwh=600.0, charge_efficiency=0.0)

    def test_rejects_efficiency_above_one(self):
        with pytest.raises(ValidationError):
            BatterySpec(rated_power_mw=100.0, energy_capacity_mwh=600.0, discharge_efficiency=1.5)

    def test_rejects_negative_aux_power(self):
        with pytest.raises(ValidationError):
            BatterySpec(rated_power_mw=100.0, energy_capacity_mwh=600.0, auxiliary_power_mw=-0.1)

    def test_rejects_negative_self_discharge(self):
        with pytest.raises(ValidationError):
            BatterySpec(
                rated_power_mw=100.0,
                energy_capacity_mwh=600.0,
                self_discharge_rate_per_hour=-0.001,
            )

    def test_rejects_min_soc_gt_max(self):
        with pytest.raises(ValidationError):
            BatterySpec(
                rated_power_mw=100.0,
                energy_capacity_mwh=600.0,
                min_soc_percent=90.0,
                max_soc_percent=10.0,
            )

    def test_min_energy_property(self):
        spec = BatterySpec(rated_power_mw=100.0, energy_capacity_mwh=600.0, min_soc_percent=10.0)
        assert spec.min_energy_mwh == pytest.approx(60.0)

    def test_max_energy_property(self):
        spec = BatterySpec(rated_power_mw=100.0, energy_capacity_mwh=600.0, max_soc_percent=90.0)
        assert spec.max_energy_mwh == pytest.approx(540.0)


class TestBatteryTelemetry:
    def test_valid_creation(self):
        t = BatteryTelemetry(timestamp=datetime(2026, 1, 1))
        assert t.current_energy_mwh is None
        assert t.is_available is True

    def test_rejects_invalid_soc(self):
        with pytest.raises(ValidationError):
            BatteryTelemetry(timestamp=datetime(2026, 1, 1), current_soc_percent=150.0)

    def test_get_energy_from_soc(self):
        spec = BatterySpec(rated_power_mw=100.0, energy_capacity_mwh=600.0)
        t = BatteryTelemetry(timestamp=datetime(2026, 1, 1), current_soc_percent=50.0)
        assert t.get_energy_from_soc(spec) == pytest.approx(300.0)

    def test_get_energy_from_soc_none(self):
        spec = BatterySpec(rated_power_mw=100.0, energy_capacity_mwh=600.0)
        t = BatteryTelemetry(timestamp=datetime(2026, 1, 1))
        assert t.get_energy_from_soc(spec) is None

    def test_get_soc_from_energy(self):
        spec = BatterySpec(rated_power_mw=100.0, energy_capacity_mwh=600.0)
        t = BatteryTelemetry(timestamp=datetime(2026, 1, 1), current_energy_mwh=300.0)
        assert t.get_soc_from_energy(spec) == pytest.approx(50.0)

    def test_get_soc_from_energy_none(self):
        spec = BatterySpec(rated_power_mw=100.0, energy_capacity_mwh=600.0)
        t = BatteryTelemetry(timestamp=datetime(2026, 1, 1))
        assert t.get_soc_from_energy(spec) is None


class TestBatteryState:
    def test_valid_creation(self):
        state = BatteryState(
            timestamp=datetime(2026, 1, 1),
            energy_mwh=300.0,
            soc_percent=50.0,
            power_mw=0.0,
            operating_mode="idle",
            is_available=True,
            estimated_values=set(),
            quality_flags=set(),
        )
        assert state.energy_mwh == 300.0

    def test_rejects_invalid_soc(self):
        with pytest.raises(ValidationError):
            BatteryState(
                timestamp=datetime(2026, 1, 1),
                energy_mwh=300.0,
                soc_percent=-1.0,
                power_mw=0.0,
                operating_mode="idle",
                is_available=True,
                estimated_values=set(),
                quality_flags=set(),
            )

    def test_has_quality_issues_false(self):
        state = BatteryState(
            timestamp=datetime(2026, 1, 1),
            energy_mwh=300.0,
            soc_percent=50.0,
            power_mw=0.0,
            operating_mode="idle",
            is_available=True,
            estimated_values=set(),
            quality_flags=set(),
        )
        assert state.has_quality_issues is False

    def test_has_quality_issues_true(self):
        state = BatteryState(
            timestamp=datetime(2026, 1, 1),
            energy_mwh=300.0,
            soc_percent=50.0,
            power_mw=0.0,
            operating_mode="idle",
            is_available=True,
            estimated_values=set(),
            quality_flags={"energy_clamped"},
        )
        assert state.has_quality_issues is True

    def test_has_estimated_values(self):
        state = BatteryState(
            timestamp=datetime(2026, 1, 1),
            energy_mwh=300.0,
            soc_percent=50.0,
            power_mw=0.0,
            operating_mode="idle",
            is_available=True,
            estimated_values={"energy_mwh"},
            quality_flags=set(),
        )
        assert state.has_estimated_values is True

    def test_has_no_estimated_values(self):
        state = BatteryState(
            timestamp=datetime(2026, 1, 1),
            energy_mwh=300.0,
            soc_percent=50.0,
            power_mw=0.0,
            operating_mode="idle",
            is_available=True,
            estimated_values=set(),
            quality_flags=set(),
        )
        assert state.has_estimated_values is False
