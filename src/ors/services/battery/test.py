"""Comprehensive unit tests for battery_management module.

This test suite covers all functions and classes in battery_management.py:
- BatteryParams class
- LossBreakdown class
- clamp function
- compute_losses function
- step_energy function
- Configuration management functions
- CSV logging functions
"""

import csv
import os
import tempfile

import pytest

# Import handling for both mypy (relative) and direct execution (absolute)
try:
    # When run from project root with mypy or as a module
    from .battery_management import (
        BatteryParams,
        LossBreakdown,
        clamp,
        compute_losses,
        create_log_entry,
        step_energy,
        write_simulation_csv,
    )
except (ImportError, ModuleNotFoundError):
    # When run directly from this directory
    from battery_management import (  # type: ignore[import-not-found,no-redef]
        BatteryParams,
        LossBreakdown,
        clamp,
        compute_losses,
        create_log_entry,
        step_energy,
        write_simulation_csv,
    )


class TestBatteryParams:
    """Test BatteryParams class."""

    def test_default_initialization(self) -> None:
        """Test creating BatteryParams with default values."""
        params = BatteryParams()

        assert params.p_rated_mw == 100.0
        assert params.eta_ch == 0.97
        assert params.eta_dis == 0.97
        assert params.a_aux == 0.005
        assert params.r_sd_per_hour == 0.0005
        assert params.e_duration_hours == 3.0
        assert params.e_min_frac == 0.10
        assert params.e_max_frac == 0.90

    def test_custom_initialization(self) -> None:
        """Test creating BatteryParams with custom values."""
        params = BatteryParams(
            p_rated_mw=200.0,
            eta_ch=0.95,
            eta_dis=0.95,
            a_aux=0.01,
            r_sd_per_hour=0.001,
            e_duration_hours=4.0,
            e_min_frac=0.05,
            e_max_frac=0.95,
        )

        assert params.p_rated_mw == 200.0
        assert params.eta_ch == 0.95
        assert params.eta_dis == 0.95
        assert params.a_aux == 0.01
        assert params.r_sd_per_hour == 0.001
        assert params.e_duration_hours == 4.0
        assert params.e_min_frac == 0.05
        assert params.e_max_frac == 0.95

    def test_computed_properties(self) -> None:
        """Test computed properties."""
        params = BatteryParams(
            p_rated_mw=100.0,
            a_aux=0.005,
            e_duration_hours=3.0,
            e_min_frac=0.1,
            e_max_frac=0.9,
        )

        assert params.p_aux_mw == 0.5  # 0.005 * 100
        assert params.e_cap_mwh == 300.0  # 100 * 3
        assert params.e_min_mwh == 30.0  # 0.1 * 300
        assert params.e_max_mwh == 270.0  # 0.9 * 300

    def test_immutability(self) -> None:
        """Test that BatteryParams instances are immutable."""
        params = BatteryParams()

        with pytest.raises(AttributeError, match="BatteryParams is immutable"):
            params.p_rated_mw = 200.0

        with pytest.raises(AttributeError, match="BatteryParams is immutable"):
            params.eta_ch = 0.95

    def test_repr(self) -> None:
        """Test string representation."""
        params = BatteryParams(p_rated_mw=50.0, eta_ch=0.95)
        repr_str = repr(params)

        assert "BatteryParams(" in repr_str
        assert "p_rated_mw=50.0" in repr_str
        assert "eta_ch=0.95" in repr_str


class TestBatteryParamsValidation:
    """Test BatteryParams validation with invalid inputs."""

    def test_negative_rated_power(self) -> None:
        """Test error when p_rated_mw is negative or zero."""
        with pytest.raises(ValueError, match="p_rated_mw must be > 0"):
            BatteryParams(p_rated_mw=-100.0)

        with pytest.raises(ValueError, match="p_rated_mw must be > 0"):
            BatteryParams(p_rated_mw=0.0)

    def test_invalid_charging_efficiency(self) -> None:
        """Test error when eta_ch is invalid."""
        with pytest.raises(ValueError, match="eta_ch must be in \\(0, 1\\]"):
            BatteryParams(eta_ch=0.0)

        with pytest.raises(ValueError, match="eta_ch must be in \\(0, 1\\]"):
            BatteryParams(eta_ch=-0.1)

        with pytest.raises(ValueError, match="eta_ch must be in \\(0, 1\\]"):
            BatteryParams(eta_ch=1.5)

    def test_invalid_discharging_efficiency(self) -> None:
        """Test error when eta_dis is invalid."""
        with pytest.raises(ValueError, match="eta_dis must be in \\(0, 1\\]"):
            BatteryParams(eta_dis=0.0)

        with pytest.raises(ValueError, match="eta_dis must be in \\(0, 1\\]"):
            BatteryParams(eta_dis=-0.1)

        with pytest.raises(ValueError, match="eta_dis must be in \\(0, 1\\]"):
            BatteryParams(eta_dis=1.1)

    def test_negative_auxiliary_power(self) -> None:
        """Test error when a_aux is negative."""
        with pytest.raises(ValueError, match="a_aux must be >= 0"):
            BatteryParams(a_aux=-0.001)

    def test_negative_self_discharge_rate(self) -> None:
        """Test error when r_sd_per_hour is negative."""
        with pytest.raises(ValueError, match="r_sd_per_hour must be >= 0"):
            BatteryParams(r_sd_per_hour=-0.001)

    def test_invalid_duration(self) -> None:
        """Test error when e_duration_hours is negative or zero."""
        with pytest.raises(ValueError, match="e_duration_hours must be > 0"):
            BatteryParams(e_duration_hours=0.0)

        with pytest.raises(ValueError, match="e_duration_hours must be > 0"):
            BatteryParams(e_duration_hours=-1.0)

    def test_invalid_energy_fractions(self) -> None:
        """Test error when energy fractions are outside [0,1]."""
        with pytest.raises(ValueError, match="e_min_frac must be in \\[0, 1\\]"):
            BatteryParams(e_min_frac=-0.1)

        with pytest.raises(ValueError, match="e_min_frac must be in \\[0, 1\\]"):
            BatteryParams(e_min_frac=1.1)

        with pytest.raises(ValueError, match="e_max_frac must be in \\[0, 1\\]"):
            BatteryParams(e_max_frac=-0.1)

        with pytest.raises(ValueError, match="e_max_frac must be in \\[0, 1\\]"):
            BatteryParams(e_max_frac=1.1)

    def test_min_greater_than_max_fraction(self) -> None:
        """Test error when e_min_frac > e_max_frac."""
        with pytest.raises(ValueError, match="e_min_frac must be <= e_max_frac"):
            BatteryParams(e_min_frac=0.8, e_max_frac=0.2)

    def test_boundary_conditions(self) -> None:
        """Test that boundary values are accepted."""
        # These should work without errors
        BatteryParams(eta_ch=1.0, eta_dis=1.0)  # Perfect efficiency
        BatteryParams(eta_ch=0.01, eta_dis=0.01)  # Very low efficiency
        BatteryParams(e_min_frac=0.0, e_max_frac=1.0)  # Full range
        BatteryParams(e_min_frac=0.5, e_max_frac=0.5)  # Equal min/max
        BatteryParams(a_aux=0.0, r_sd_per_hour=0.0)  # No losses


class TestLossBreakdown:
    """Test LossBreakdown class."""

    def test_initialization(self) -> None:
        """Test creating LossBreakdown."""
        losses = LossBreakdown(
            loss_charge_ineff_mwh=1.0,
            loss_discharge_ineff_mwh=2.0,
            loss_aux_mwh=0.5,
            loss_self_discharge_mwh=0.3,
        )

        assert losses.loss_charge_ineff_mwh == 1.0
        assert losses.loss_discharge_ineff_mwh == 2.0
        assert losses.loss_aux_mwh == 0.5
        assert losses.loss_self_discharge_mwh == 0.3

    def test_total_loss_property(self) -> None:
        """Test total_loss_mwh property."""
        losses = LossBreakdown(
            loss_charge_ineff_mwh=1.0,
            loss_discharge_ineff_mwh=2.0,
            loss_aux_mwh=0.5,
            loss_self_discharge_mwh=0.3,
        )

        expected_total = 1.0 + 2.0 + 0.5 + 0.3
        assert losses.total_loss_mwh == expected_total

    def test_immutability(self) -> None:
        """Test that LossBreakdown instances are immutable."""
        losses = LossBreakdown(
            loss_charge_ineff_mwh=1.0,
            loss_discharge_ineff_mwh=2.0,
            loss_aux_mwh=0.5,
            loss_self_discharge_mwh=0.3,
        )

        with pytest.raises(AttributeError, match="LossBreakdown is immutable"):
            losses.loss_charge_ineff_mwh = 1.5

    def test_repr(self) -> None:
        """Test string representation."""
        losses = LossBreakdown(
            loss_charge_ineff_mwh=1.0,
            loss_discharge_ineff_mwh=2.0,
            loss_aux_mwh=0.5,
            loss_self_discharge_mwh=0.3,
        )
        repr_str = repr(losses)

        assert "LossBreakdown(" in repr_str
        assert "loss_charge_ineff_mwh=1.0" in repr_str


class TestClamp:
    """Test clamp function."""

    def test_value_within_bounds(self) -> None:
        """Test clamping when value is within bounds."""
        assert clamp(5.0, 0.0, 10.0) == 5.0
        assert clamp(2.5, 1.0, 3.0) == 2.5

    def test_value_below_lower_bound(self) -> None:
        """Test clamping when value is below lower bound."""
        assert clamp(-5.0, 0.0, 10.0) == 0.0
        assert clamp(0.5, 1.0, 3.0) == 1.0

    def test_value_above_upper_bound(self) -> None:
        """Test clamping when value is above upper bound."""
        assert clamp(15.0, 0.0, 10.0) == 10.0
        assert clamp(5.0, 1.0, 3.0) == 3.0

    def test_value_equals_bounds(self) -> None:
        """Test clamping when value equals bounds."""
        assert clamp(0.0, 0.0, 10.0) == 0.0
        assert clamp(10.0, 0.0, 10.0) == 10.0

    def test_negative_values(self) -> None:
        """Test clamping with negative values."""
        assert clamp(-5.0, -10.0, -1.0) == -5.0
        assert clamp(-15.0, -10.0, -1.0) == -10.0
        assert clamp(0.0, -10.0, -1.0) == -1.0


class TestComputeLosses:
    """Test compute_losses function."""

    def test_charging_only_losses(self) -> None:
        """Test losses when only charging."""
        params = BatteryParams(eta_ch=0.9, p_rated_mw=100.0, a_aux=0.01, r_sd_per_hour=0.001)

        losses = compute_losses(
            e_prev_mwh=150.0,
            p_grid_mw=50.0,
            p_sol_mw=30.0,
            p_dis_mw=0.0,
            params=params,
            dt_hours=1.0,
        )

        # Charging inefficiency: (1 - 0.9) * (50 + 30) * 1 = 8.0
        assert losses.loss_charge_ineff_mwh == pytest.approx(8.0, abs=1e-10)

        # No discharge
        assert losses.loss_discharge_ineff_mwh == pytest.approx(0.0, abs=1e-10)

        # Auxiliary: 0.01 * 100 * 1 = 1.0
        assert losses.loss_aux_mwh == pytest.approx(1.0, abs=1e-10)

        # Self-discharge: 150 * 0.001 * 1 = 0.15
        assert losses.loss_self_discharge_mwh == pytest.approx(0.15, abs=1e-10)

    def test_discharging_only_losses(self) -> None:
        """Test losses when only discharging."""
        params = BatteryParams(eta_dis=0.9, p_rated_mw=100.0, a_aux=0.01, r_sd_per_hour=0.001)

        losses = compute_losses(
            e_prev_mwh=150.0,
            p_grid_mw=0.0,
            p_sol_mw=0.0,
            p_dis_mw=45.0,
            params=params,
            dt_hours=1.0,
        )

        # No charging
        assert losses.loss_charge_ineff_mwh == pytest.approx(0.0, abs=1e-10)

        # Discharge inefficiency: 45 * 1 * (1/0.9 - 1) = 45 * (10/9 - 1) = 5.0
        expected_discharge_loss = 45.0 * (1.0 / 0.9 - 1.0)
        assert losses.loss_discharge_ineff_mwh == pytest.approx(expected_discharge_loss, abs=1e-10)

        # Auxiliary: 0.01 * 100 * 1 = 1.0
        assert losses.loss_aux_mwh == pytest.approx(1.0, abs=1e-10)

        # Self-discharge: 150 * 0.001 * 1 = 0.15
        assert losses.loss_self_discharge_mwh == pytest.approx(0.15, abs=1e-10)

    def test_zero_time_step_error(self) -> None:
        """Test error when dt_hours is zero."""
        params = BatteryParams()

        with pytest.raises(ValueError, match="dt_hours must be > 0"):
            compute_losses(
                e_prev_mwh=150.0,
                p_grid_mw=50.0,
                p_sol_mw=30.0,
                p_dis_mw=20.0,
                params=params,
                dt_hours=0.0,
            )

    def test_negative_time_step_error(self) -> None:
        """Test error when dt_hours is negative."""
        params = BatteryParams()

        with pytest.raises(ValueError, match="dt_hours must be > 0"):
            compute_losses(
                e_prev_mwh=150.0,
                p_grid_mw=50.0,
                p_sol_mw=30.0,
                p_dis_mw=20.0,
                params=params,
                dt_hours=-0.5,
            )


class TestStepEnergy:
    """Test step_energy function."""

    def test_charging_step(self) -> None:
        """Test energy step with charging."""
        params = BatteryParams(
            eta_ch=0.9,
            eta_dis=0.9,
            p_rated_mw=100.0,
            a_aux=0.01,
            r_sd_per_hour=0.001,
        )

        e_next = step_energy(
            e_prev_mwh=150.0,
            p_grid_mw=50.0,
            p_sol_mw=30.0,
            p_dis_mw=0.0,
            params=params,
            dt_hours=1.0,
        )

        # Expected: 150 + 0.9*(50+30)*1 - 0 - 1.0*1 - 150*0.001*1
        # = 150 + 72 - 0 - 1 - 0.15 = 220.85
        expected = 150.0 + 0.9 * 80.0 - 1.0 - 150.0 * 0.001
        assert e_next == pytest.approx(expected, abs=1e-10)

    def test_discharging_step(self) -> None:
        """Test energy step with discharging."""
        params = BatteryParams(
            eta_ch=0.9,
            eta_dis=0.9,
            p_rated_mw=100.0,
            a_aux=0.01,
            r_sd_per_hour=0.001,
        )

        e_next = step_energy(
            e_prev_mwh=150.0,
            p_grid_mw=0.0,
            p_sol_mw=0.0,
            p_dis_mw=45.0,
            params=params,
            dt_hours=1.0,
        )

        # Expected: 150 + 0 - 45*1/0.9 - 1.0*1 - 150*0.001*1
        # = 150 - 50 - 1 - 0.15 = 98.85
        expected = 150.0 - 45.0 / 0.9 - 1.0 - 150.0 * 0.001
        assert e_next == pytest.approx(expected, abs=1e-10)

    def test_bounds_enforcement(self) -> None:
        """Test energy bounds enforcement."""
        params = BatteryParams(
            e_min_frac=0.1,  # e_min = 30 MWh
            e_max_frac=0.9,  # e_max = 270 MWh
            eta_ch=1.0,  # Perfect efficiency for easy calculation
            eta_dis=1.0,
            a_aux=0.0,  # No auxiliary losses
            r_sd_per_hour=0.0,  # No self-discharge
        )

        # Test upper bound clamping
        e_high = step_energy(
            e_prev_mwh=260.0,
            p_grid_mw=50.0,  # Would add 50 MWh -> 310 MWh
            p_sol_mw=0.0,
            p_dis_mw=0.0,
            params=params,
            dt_hours=1.0,
            enforce_bounds=True,
        )
        assert e_high == params.e_max_mwh  # 270 MWh

        # Test lower bound clamping
        e_low = step_energy(
            e_prev_mwh=50.0,
            p_grid_mw=0.0,
            p_sol_mw=0.0,
            p_dis_mw=30.0,  # Would remove 30 MWh -> 20 MWh
            params=params,
            dt_hours=1.0,
            enforce_bounds=True,
        )
        assert e_low == params.e_min_mwh  # 30 MWh

    def test_no_bounds_enforcement(self) -> None:
        """Test when bounds enforcement is disabled."""
        params = BatteryParams(
            e_min_frac=0.1,
            e_max_frac=0.9,
            eta_ch=1.0,
            eta_dis=1.0,
            a_aux=0.0,
            r_sd_per_hour=0.0,
        )

        e_next = step_energy(
            e_prev_mwh=260.0,
            p_grid_mw=50.0,  # Would add 50 MWh -> 310 MWh
            p_sol_mw=0.0,
            p_dis_mw=0.0,
            params=params,
            dt_hours=1.0,
            enforce_bounds=False,
        )
        assert e_next == 310.0  # No clamping

    def test_zero_time_step_error(self) -> None:
        """Test error when dt_hours is zero."""
        params = BatteryParams()

        with pytest.raises(ValueError, match="dt_hours must be > 0"):
            step_energy(
                e_prev_mwh=150.0,
                p_grid_mw=50.0,
                p_sol_mw=30.0,
                p_dis_mw=20.0,
                params=params,
                dt_hours=0.0,
            )


class TestCSVLogging:
    """Test CSV logging functions."""

    @pytest.fixture
    def sample_losses(self) -> LossBreakdown:
        """Create sample losses for testing."""
        return LossBreakdown(
            loss_charge_ineff_mwh=1.5,
            loss_discharge_ineff_mwh=0.8,
            loss_aux_mwh=0.25,
            loss_self_discharge_mwh=0.075,
        )

    def test_create_log_entry(self, sample_losses: LossBreakdown) -> None:
        """Test creating a log entry."""
        entry = create_log_entry(
            step=0,
            t_hours=0.25,
            dt_hours=0.25,
            p_grid_mw=40.0,
            p_sol_mw=20.0,
            p_dis_mw=0.0,
            e_prev_mwh=150.0,
            e_next_mwh=165.5,
            losses=sample_losses,
            eta_ch=0.97,
        )

        assert entry["simulation_step"] == 0
        assert entry["elapsed_time_hours"] == 0.25
        assert entry["time_step_hours"] == 0.25
        assert entry["grid_power_mw"] == 40.0
        assert entry["solar_power_mw"] == 20.0
        assert entry["discharge_power_mw"] == 0.0
        assert entry["energy_before_mwh"] == 150.0
        assert entry["energy_after_mwh"] == 165.5

        # Check calculated energy flows
        expected_e_in_grid = 40.0 * 0.97 * 0.25
        expected_e_in_sol = 20.0 * 0.97 * 0.25
        assert entry["energy_in_from_grid_mwh"] == pytest.approx(expected_e_in_grid)
        assert entry["energy_in_from_solar_mwh"] == pytest.approx(expected_e_in_sol)
        assert entry["energy_in_total_mwh"] == pytest.approx(expected_e_in_grid + expected_e_in_sol)

        # Check loss values
        assert entry["loss_charging_inefficiency_mwh"] == 1.5
        assert entry["loss_total_mwh"] == sample_losses.total_loss_mwh

    def test_create_log_entry_with_timestamp(self, sample_losses: LossBreakdown) -> None:
        """Test creating a log entry with timestamp."""
        timestamp = "2026-02-22T10:30:00Z"
        entry = create_log_entry(
            step=5,
            t_hours=1.25,
            dt_hours=0.25,
            p_grid_mw=0.0,
            p_sol_mw=0.0,
            p_dis_mw=30.0,
            e_prev_mwh=200.0,
            e_next_mwh=175.0,
            losses=sample_losses,
            eta_ch=0.97,
            timestamp_iso=timestamp,
        )

        assert entry["timestamp_iso"] == timestamp
        assert entry["simulation_step"] == 5

    def test_write_simulation_csv(self, sample_losses: LossBreakdown) -> None:
        """Test writing simulation data to CSV."""
        logs = [
            create_log_entry(
                step=0,
                t_hours=0.0,
                dt_hours=0.25,
                p_grid_mw=50.0,
                p_sol_mw=0.0,
                p_dis_mw=0.0,
                e_prev_mwh=150.0,
                e_next_mwh=162.0,
                losses=sample_losses,
                eta_ch=0.97,
            ),
            create_log_entry(
                step=1,
                t_hours=0.25,
                dt_hours=0.25,
                p_grid_mw=0.0,
                p_sol_mw=30.0,
                p_dis_mw=0.0,
                e_prev_mwh=162.0,
                e_next_mwh=169.0,
                losses=sample_losses,
                eta_ch=0.97,
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            write_simulation_csv(logs, temp_path)

            # Verify CSV was written correctly
            with open(temp_path) as f:
                reader = csv.DictReader(f)
                rows = list(reader)

                assert len(rows) == 2
                assert rows[0]["simulation_step"] == "0"
                assert rows[0]["grid_power_mw"] == "50.0"
                assert rows[1]["simulation_step"] == "1"
                assert rows[1]["solar_power_mw"] == "30.0"

        finally:
            os.unlink(temp_path)

    def test_write_simulation_csv_empty_logs(self) -> None:
        """Test error when writing empty logs."""
        with pytest.raises(ValueError, match="Cannot write empty logs to CSV"):
            write_simulation_csv([], "test.csv")

    def test_write_simulation_csv_with_timestamp(self, sample_losses: LossBreakdown) -> None:
        """Test writing CSV with timestamp column."""
        logs = [
            create_log_entry(
                step=0,
                t_hours=0.0,
                dt_hours=0.25,
                p_grid_mw=50.0,
                p_sol_mw=0.0,
                p_dis_mw=0.0,
                e_prev_mwh=150.0,
                e_next_mwh=162.0,
                losses=sample_losses,
                eta_ch=0.97,
                timestamp_iso="2026-02-22T10:00:00Z",
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

        try:
            write_simulation_csv(logs, temp_path)

            # Verify timestamp column is present
            with open(temp_path) as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                rows = list(reader)

                assert fieldnames is not None
                assert "timestamp_iso" in fieldnames
                assert rows[0]["timestamp_iso"] == "2026-02-22T10:00:00Z"

        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    # Run all tests with detailed output
    pytest.main([__file__, "-v", "--tb=short"])
