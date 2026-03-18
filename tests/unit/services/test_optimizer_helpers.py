"""Tests for optimizer helper functions (get_cycles_used_today, extract_optimizer_initial_state)."""

import pytest
from src.ors.services.optimizer.optimizer import (
    extract_optimizer_initial_state,
    get_cycles_used_today,
)

_E_CAP = 600.0  # Local constant for tests (matches default battery capacity)


class TestGetCyclesUsedToday:
    def test_none_returns_zero(self):
        assert get_cycles_used_today(None) == 0

    def test_empty_list_returns_zero(self):
        assert get_cycles_used_today([]) == 0

    def test_no_cycle_events(self):
        logs = [{"action": "charge"}, {"action": "discharge"}]
        assert get_cycles_used_today(logs) == 0

    def test_counts_cycle_events(self):
        logs = [
            {"cycle": 1, "action": "discharge"},
            {"cycle": 0, "action": "idle"},
            {"cycle": 1, "action": "discharge"},
        ]
        assert get_cycles_used_today(logs) == 2


class TestExtractOptimizerInitialState:
    def test_none_returns_defaults(self):
        energy, mode, cycles = extract_optimizer_initial_state(None, e_cap_mwh=_E_CAP)
        assert energy == pytest.approx(_E_CAP * 0.5)
        assert mode == "idle"
        assert cycles == 0

    def test_with_battery_state(self):
        class FakeState:
            energy_mwh = 200.0
            operating_mode = "charging"

        energy, mode, cycles = extract_optimizer_initial_state(FakeState(), e_cap_mwh=_E_CAP)
        assert energy == pytest.approx(200.0)
        assert mode == "charging"
        assert cycles == 0

    def test_missing_attributes_uses_defaults(self):
        class MinimalState:
            pass

        energy, mode, cycles = extract_optimizer_initial_state(MinimalState(), e_cap_mwh=_E_CAP)
        assert energy == pytest.approx(_E_CAP * 0.5)
        assert mode == "idle"
