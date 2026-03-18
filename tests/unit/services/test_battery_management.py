"""Tests for battery_management module - covers config loading, CSV, step_energy, validation."""

import pytest
from src.ors.services.battery.battery_management import (
    BatteryParams,
    LossBreakdown,
    clamp,
    compute_losses,
    create_log_entry,
    step_energy,
    write_simulation_csv,
)

# ---------------------------------------------------------------------------
# BatteryParams validation
# ---------------------------------------------------------------------------


class TestBatteryParamsValidation:
    def test_rejects_zero_power(self):
        with pytest.raises(ValueError, match="p_rated_mw"):
            BatteryParams(p_rated_mw=0)

    def test_rejects_negative_power(self):
        with pytest.raises(ValueError, match="p_rated_mw"):
            BatteryParams(p_rated_mw=-10)

    def test_rejects_eta_ch_zero(self):
        with pytest.raises(ValueError, match="eta_ch"):
            BatteryParams(eta_ch=0)

    def test_rejects_eta_ch_above_one(self):
        with pytest.raises(ValueError, match="eta_ch"):
            BatteryParams(eta_ch=1.1)

    def test_rejects_eta_dis_zero(self):
        with pytest.raises(ValueError, match="eta_dis"):
            BatteryParams(eta_dis=0)

    def test_rejects_negative_a_aux(self):
        with pytest.raises(ValueError, match="a_aux"):
            BatteryParams(a_aux=-0.01)

    def test_rejects_negative_r_sd(self):
        with pytest.raises(ValueError, match="r_sd_per_hour"):
            BatteryParams(r_sd_per_hour=-0.001)

    def test_rejects_zero_duration(self):
        with pytest.raises(ValueError, match="e_duration_hours"):
            BatteryParams(e_duration_hours=0)

    def test_rejects_e_min_frac_above_one(self):
        with pytest.raises(ValueError, match="e_min_frac"):
            BatteryParams(e_min_frac=1.1)

    def test_rejects_e_min_frac_negative(self):
        with pytest.raises(ValueError, match="e_min_frac"):
            BatteryParams(e_min_frac=-0.1)

    def test_rejects_e_max_frac_negative(self):
        with pytest.raises(ValueError, match="e_max_frac"):
            BatteryParams(e_max_frac=-0.1)

    def test_rejects_min_gt_max(self):
        with pytest.raises(ValueError, match="e_min_frac must be <= e_max_frac"):
            BatteryParams(e_min_frac=0.9, e_max_frac=0.1)


class TestBatteryParamsImmutability:
    def test_frozen_after_init(self):
        params = BatteryParams()
        with pytest.raises(AttributeError, match="immutable"):
            params.p_rated_mw = 200.0


class TestBatteryParamsProperties:
    def test_p_aux_mw(self):
        params = BatteryParams(p_rated_mw=100.0, a_aux=0.005)
        assert params.p_aux_mw == pytest.approx(0.5)

    def test_e_cap_mwh(self):
        params = BatteryParams(p_rated_mw=100.0, e_duration_hours=3.0)
        assert params.e_cap_mwh == pytest.approx(300.0)

    def test_e_min_mwh(self):
        params = BatteryParams(p_rated_mw=100.0, e_duration_hours=3.0, e_min_frac=0.1)
        assert params.e_min_mwh == pytest.approx(30.0)

    def test_e_max_mwh(self):
        params = BatteryParams(p_rated_mw=100.0, e_duration_hours=3.0, e_max_frac=0.9)
        assert params.e_max_mwh == pytest.approx(270.0)


# ---------------------------------------------------------------------------
# LossBreakdown
# ---------------------------------------------------------------------------


class TestLossBreakdown:
    def test_immutable(self):
        lb = LossBreakdown(
            loss_charge_ineff_mwh=1.0,
            loss_discharge_ineff_mwh=2.0,
            loss_aux_mwh=0.5,
            loss_self_discharge_mwh=0.1,
        )
        with pytest.raises(AttributeError, match="immutable"):
            lb.loss_aux_mwh = 99.0

    def test_total_loss(self):
        lb = LossBreakdown(
            loss_charge_ineff_mwh=1.0,
            loss_discharge_ineff_mwh=2.0,
            loss_aux_mwh=0.5,
            loss_self_discharge_mwh=0.1,
        )
        assert lb.total_loss_mwh == pytest.approx(3.6)


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------


class TestClamp:
    def test_within_range(self):
        assert clamp(5.0, 0.0, 10.0) == 5.0

    def test_below_low(self):
        assert clamp(-1.0, 0.0, 10.0) == 0.0

    def test_above_high(self):
        assert clamp(15.0, 0.0, 10.0) == 10.0


# ---------------------------------------------------------------------------
# step_energy
# ---------------------------------------------------------------------------


class TestStepEnergy:
    def test_idle_step(self):
        params = BatteryParams()
        e = step_energy(
            e_prev_mwh=150.0,
            p_grid_mw=0,
            p_sol_mw=0,
            p_dis_mw=0,
            params=params,
            dt_hours=0.25,
        )
        # Should decrease due to aux and self-discharge
        assert e < 150.0

    def test_charge_increases_energy(self):
        params = BatteryParams()
        e = step_energy(
            e_prev_mwh=150.0,
            p_grid_mw=50.0,
            p_sol_mw=0,
            p_dis_mw=0,
            params=params,
            dt_hours=0.25,
        )
        assert e > 150.0

    def test_discharge_decreases_energy(self):
        params = BatteryParams()
        e = step_energy(
            e_prev_mwh=200.0,
            p_grid_mw=0,
            p_sol_mw=0,
            p_dis_mw=50.0,
            params=params,
            dt_hours=0.25,
        )
        assert e < 200.0

    def test_rejects_zero_dt(self):
        params = BatteryParams()
        with pytest.raises(ValueError, match="dt_hours"):
            step_energy(
                e_prev_mwh=150.0,
                p_grid_mw=0,
                p_sol_mw=0,
                p_dis_mw=0,
                params=params,
                dt_hours=0,
            )

    def test_enforce_bounds_clamps(self):
        params = BatteryParams(p_rated_mw=100.0, e_duration_hours=3.0, e_min_frac=0.1)
        # Try to discharge below e_min
        e = step_energy(
            e_prev_mwh=35.0,
            p_grid_mw=0,
            p_sol_mw=0,
            p_dis_mw=100.0,
            params=params,
            dt_hours=0.25,
            enforce_bounds=True,
        )
        assert e >= params.e_min_mwh

    def test_no_enforce_bounds(self):
        params = BatteryParams(p_rated_mw=100.0, e_duration_hours=3.0, e_min_frac=0.1)
        e = step_energy(
            e_prev_mwh=35.0,
            p_grid_mw=0,
            p_sol_mw=0,
            p_dis_mw=100.0,
            params=params,
            dt_hours=0.25,
            enforce_bounds=False,
        )
        # Without bounds enforcement, can go below e_min
        assert e < params.e_min_mwh


# ---------------------------------------------------------------------------
# compute_losses
# ---------------------------------------------------------------------------


class TestComputeLosses:
    def test_rejects_zero_dt(self):
        params = BatteryParams()
        with pytest.raises(ValueError, match="dt_hours"):
            compute_losses(
                e_prev_mwh=150.0,
                p_grid_mw=0,
                p_sol_mw=0,
                p_dis_mw=0,
                params=params,
                dt_hours=0,
            )

    def test_idle_has_aux_and_self_discharge(self):
        params = BatteryParams()
        lb = compute_losses(
            e_prev_mwh=150.0,
            p_grid_mw=0,
            p_sol_mw=0,
            p_dis_mw=0,
            params=params,
            dt_hours=0.25,
        )
        assert lb.loss_aux_mwh > 0
        assert lb.loss_self_discharge_mwh > 0
        assert lb.loss_charge_ineff_mwh == 0.0
        assert lb.loss_discharge_ineff_mwh == 0.0


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CSV logging
# ---------------------------------------------------------------------------


class TestCSVLogging:
    def _make_log_entry(self):
        params = BatteryParams()
        lb = compute_losses(
            e_prev_mwh=150.0,
            p_grid_mw=50.0,
            p_sol_mw=0,
            p_dis_mw=0,
            params=params,
            dt_hours=0.25,
        )
        return create_log_entry(
            step=0,
            t_hours=0.0,
            dt_hours=0.25,
            p_grid_mw=50.0,
            p_sol_mw=0.0,
            p_dis_mw=0.0,
            e_prev_mwh=150.0,
            e_next_mwh=162.0,
            losses=lb,
            eta_ch=params.eta_ch,
        )

    def test_create_log_entry(self):
        entry = self._make_log_entry()
        assert entry["simulation_step"] == 0
        assert entry["grid_power_mw"] == 50.0
        assert "energy_before_mwh" in entry

    def test_create_log_entry_with_timestamp(self):
        params = BatteryParams()
        lb = compute_losses(
            e_prev_mwh=150.0,
            p_grid_mw=0,
            p_sol_mw=0,
            p_dis_mw=0,
            params=params,
            dt_hours=0.25,
        )
        entry = create_log_entry(
            step=0,
            t_hours=0.0,
            dt_hours=0.25,
            p_grid_mw=0,
            p_sol_mw=0,
            p_dis_mw=0,
            e_prev_mwh=150.0,
            e_next_mwh=149.0,
            losses=lb,
            eta_ch=params.eta_ch,
            timestamp_iso="2026-01-01T00:00:00",
        )
        assert entry["timestamp_iso"] == "2026-01-01T00:00:00"

    def test_write_simulation_csv(self, tmp_path):
        entry = self._make_log_entry()
        csv_path = tmp_path / "out.csv"
        write_simulation_csv([entry], csv_path)
        content = csv_path.read_text()
        assert "simulation_step" in content

    def test_write_simulation_csv_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            write_simulation_csv([])

    def test_write_csv_with_timestamp(self, tmp_path):
        params = BatteryParams()
        lb = compute_losses(
            e_prev_mwh=150.0,
            p_grid_mw=0,
            p_sol_mw=0,
            p_dis_mw=0,
            params=params,
            dt_hours=0.25,
        )
        entry = create_log_entry(
            step=0,
            t_hours=0.0,
            dt_hours=0.25,
            p_grid_mw=0,
            p_sol_mw=0,
            p_dis_mw=0,
            e_prev_mwh=150.0,
            e_next_mwh=149.0,
            losses=lb,
            eta_ch=params.eta_ch,
            timestamp_iso="2026-01-01T00:00:00",
        )
        csv_path = tmp_path / "out.csv"
        write_simulation_csv([entry], csv_path)
        content = csv_path.read_text()
        assert "timestamp_iso" in content
