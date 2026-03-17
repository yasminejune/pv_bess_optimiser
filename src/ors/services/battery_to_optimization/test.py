"""Comprehensive unit tests for battery_inference module.

This test suite covers all functions in battery_inference.py:
- load_optimizer_battery_config function
- create_optimizer_log_entries function
- export_optimizer_results function
- validate_optimizer_energy_balance function
- create_enhanced_optimizer_output function
- write_step_by_step_battery_log function

Following the same testing patterns as battery_management tests.
"""

import json
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from ors.config.optimization_config import BatteryConfiguration

# Import handling for both mypy (relative) and direct execution (absolute)
try:
    # When run from project root with mypy or as a module
    from .battery_inference import (
        battery_spec_to_params,
        create_enhanced_optimizer_output,
        create_optimizer_log_entries,
        export_optimizer_results,
        validate_optimizer_energy_balance,
        write_step_by_step_battery_log,
    )
except (ImportError, ModuleNotFoundError):
    # When run directly from this directory
    from battery_inference import (  # type: ignore[import-not-found,no-redef]
        battery_spec_to_params,
        create_enhanced_optimizer_output,
        create_optimizer_log_entries,
        export_optimizer_results,
        validate_optimizer_energy_balance,
        write_step_by_step_battery_log,
    )

# Import battery management for testing integration
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "battery"))

from battery_management import BatteryParams  # type: ignore[import-not-found]


class TestBatterySpecToParams:
    """Test battery_spec_to_params adapter function."""

    @pytest.fixture
    def sample_battery_config(self) -> BatteryConfiguration:
        """Create a sample BatteryConfiguration matching the template format."""
        return BatteryConfiguration(
            rated_power_mw=50.0,
            energy_capacity_mwh=200.0,
            min_soc_percent=10.0,
            max_soc_percent=90.0,
            max_cycles_per_day=2,
            charge_efficiency=0.97,
            discharge_efficiency=0.97,
            auxiliary_power_mw=0.3,
            self_discharge_rate_per_hour=0.0005,
        )

    def test_conversion_rated_power(self, sample_battery_config: BatteryConfiguration) -> None:
        """Test that rated_power_mw maps to p_rated_mw."""
        params = battery_spec_to_params(sample_battery_config)
        assert params.p_rated_mw == 50.0

    def test_conversion_efficiencies(self, sample_battery_config: BatteryConfiguration) -> None:
        """Test that charge/discharge efficiencies map correctly."""
        params = battery_spec_to_params(sample_battery_config)
        assert params.eta_ch == 0.97
        assert params.eta_dis == 0.97

    def test_conversion_aux_power_as_fraction(
        self, sample_battery_config: BatteryConfiguration
    ) -> None:
        """Test that auxiliary_power_mw is converted to fraction of rated power."""
        params = battery_spec_to_params(sample_battery_config)
        assert params.a_aux == pytest.approx(0.3 / 50.0)
        assert params.p_aux_mw == pytest.approx(0.3)

    def test_conversion_energy_duration(self, sample_battery_config: BatteryConfiguration) -> None:
        """Test that energy_capacity_mwh is converted to duration hours."""
        params = battery_spec_to_params(sample_battery_config)
        assert params.e_duration_hours == pytest.approx(200.0 / 50.0)
        assert params.e_cap_mwh == pytest.approx(200.0)

    def test_conversion_soc_as_fractions(self, sample_battery_config: BatteryConfiguration) -> None:
        """Test that SOC percentages are converted to fractions."""
        params = battery_spec_to_params(sample_battery_config)
        assert params.e_min_frac == pytest.approx(0.10)
        assert params.e_max_frac == pytest.approx(0.90)

    def test_conversion_self_discharge(self, sample_battery_config: BatteryConfiguration) -> None:
        """Test that self_discharge_rate_per_hour maps directly."""
        params = battery_spec_to_params(sample_battery_config)
        assert params.r_sd_per_hour == 0.0005

    def test_returns_battery_params_instance(
        self, sample_battery_config: BatteryConfiguration
    ) -> None:
        """Test that result is a BatteryParams instance."""
        params = battery_spec_to_params(sample_battery_config)
        assert isinstance(params, BatteryParams)


class TestCreateOptimizerLogEntries:
    """Test create_optimizer_log_entries function."""

    @pytest.fixture
    def sample_battery_params(self) -> BatteryParams:
        """Create sample battery parameters for testing."""
        return BatteryParams(
            p_rated_mw=100.0,
            eta_ch=0.97,
            eta_dis=0.97,
            a_aux=0.005,
            r_sd_per_hour=0.0005,
            e_duration_hours=3.0,
            e_min_frac=0.10,
            e_max_frac=0.90,
        )

    @pytest.fixture
    def sample_optimizer_results(self) -> pd.DataFrame:
        """Create sample optimizer results DataFrame."""
        data = {
            "timestamp": ["2026-02-24T10:00:00", "2026-02-24T10:15:00", "2026-02-24T10:30:00"],
            "price_intraday": [50.0, 60.0, 45.0],
            "solar_MW": [80.0, 90.0, 70.0],
            "P_grid_MW": [40.0, 0.0, 0.0],
            "P_dis_MW": [0.0, 0.0, 30.0],
            "P_sol_bat_MW": [30.0, 50.0, 0.0],
            "P_sol_sell_MW": [50.0, 40.0, 70.0],
            "E_MWh": [180.0, 200.0, 175.0],
        }
        return pd.DataFrame(data)

    def test_create_logs_success(
        self, sample_optimizer_results: pd.DataFrame, sample_battery_params: BatteryParams
    ) -> None:
        """Test successful log creation."""
        logs = create_optimizer_log_entries(
            df_results=sample_optimizer_results, params=sample_battery_params, dt_hours=0.25
        )

        assert len(logs) == 3

        # Check first log entry structure
        first_log = logs[0]
        required_fields = [
            "simulation_step",
            "elapsed_time_hours",
            "time_step_hours",
            "grid_power_mw",
            "solar_power_mw",
            "discharge_power_mw",
            "energy_before_mwh",
            "energy_after_mwh",
            "loss_total_mwh",
        ]

        for field in required_fields:
            assert field in first_log, f"Missing field: {field}"

        # Verify values for first step
        assert first_log["simulation_step"] == 0
        assert first_log["elapsed_time_hours"] == 0.0
        assert first_log["time_step_hours"] == 0.25
        assert first_log["grid_power_mw"] == 40.0
        assert first_log["solar_power_mw"] == 30.0
        assert first_log["discharge_power_mw"] == 0.0
        assert first_log["energy_after_mwh"] == 180.0

    def test_create_logs_with_start_datetime(
        self, sample_optimizer_results: pd.DataFrame, sample_battery_params: BatteryParams
    ) -> None:
        """Test log creation with start datetime."""
        start_time = datetime(2026, 2, 24, 10, 0, 0)

        logs = create_optimizer_log_entries(
            df_results=sample_optimizer_results,
            params=sample_battery_params,
            dt_hours=0.25,
            start_datetime=start_time,
        )

        # Check timestamps are correctly generated
        assert "timestamp_iso" in logs[0]
        assert logs[0]["timestamp_iso"] == "2026-02-24T10:00:00"
        assert logs[1]["timestamp_iso"] == "2026-02-24T10:15:00"

    def test_create_logs_empty_dataframe(self, sample_battery_params: BatteryParams) -> None:
        """Test with empty DataFrame."""
        empty_df = pd.DataFrame()

        logs = create_optimizer_log_entries(
            df_results=empty_df, params=sample_battery_params, dt_hours=0.25
        )

        assert logs == []

    def test_create_logs_missing_columns(self, sample_battery_params: BatteryParams) -> None:
        """Test with DataFrame missing required columns."""
        incomplete_df = pd.DataFrame(
            {
                "timestamp": ["2026-02-24T10:00:00"],
                "price_intraday": [50.0],
                # Missing required columns like P_grid_MW, E_MWh, etc.
            }
        )

        # Should handle gracefully and return empty list
        logs = create_optimizer_log_entries(
            df_results=incomplete_df, params=sample_battery_params, dt_hours=0.25
        )

        assert logs == []

    def test_create_logs_different_time_steps(
        self, sample_optimizer_results: pd.DataFrame, sample_battery_params: BatteryParams
    ) -> None:
        """Test with different time step durations."""
        logs_15_min = create_optimizer_log_entries(
            df_results=sample_optimizer_results,
            params=sample_battery_params,
            dt_hours=0.25,  # 15 minutes
        )

        logs_30_min = create_optimizer_log_entries(
            df_results=sample_optimizer_results,
            params=sample_battery_params,
            dt_hours=0.5,  # 30 minutes
        )

        # Time step should be reflected in logs
        assert logs_15_min[0]["time_step_hours"] == 0.25
        assert logs_30_min[0]["time_step_hours"] == 0.5

        # Elapsed time should scale accordingly
        assert logs_15_min[1]["elapsed_time_hours"] == 0.25
        assert logs_30_min[1]["elapsed_time_hours"] == 0.5


class TestValidateOptimizerEnergyBalance:
    """Test validate_optimizer_energy_balance function."""

    @pytest.fixture
    def perfect_balance_results(self) -> pd.DataFrame:
        """Create optimizer results with perfect energy balance."""
        # Manually constructed to have perfect energy balance
        params = BatteryParams()
        dt_hours = 0.25

        data = []
        e_current = 150.0  # Starting energy

        for step in range(3):
            if step == 0:
                p_grid, p_sol_bat, p_dis = 40.0, 20.0, 0.0
            elif step == 1:
                p_grid, p_sol_bat, p_dis = 0.0, 30.0, 0.0
            else:
                p_grid, p_sol_bat, p_dis = 0.0, 0.0, 25.0

            # Calculate next energy exactly using step_energy
            from battery_management import step_energy

            e_next = step_energy(
                e_prev_mwh=e_current,
                p_grid_mw=p_grid,
                p_sol_mw=p_sol_bat,
                p_dis_mw=p_dis,
                params=params,
                dt_hours=dt_hours,
                enforce_bounds=True,
            )

            data.append(
                {"P_grid_MW": p_grid, "P_sol_bat_MW": p_sol_bat, "P_dis_MW": p_dis, "E_MWh": e_next}
            )

            e_current = e_next

        return pd.DataFrame(data)

    @pytest.fixture
    def imperfect_balance_results(self) -> pd.DataFrame:
        """Create optimizer results with energy balance errors."""
        data = {
            "P_grid_MW": [40.0, 0.0, 0.0],
            "P_sol_bat_MW": [20.0, 30.0, 0.0],
            "P_dis_MW": [0.0, 0.0, 25.0],
            "E_MWh": [180.0, 200.5, 175.2],  # Intentionally incorrect values
        }
        return pd.DataFrame(data)

    def test_validation_perfect_balance(self, perfect_balance_results: pd.DataFrame) -> None:
        """Test validation with perfect energy balance."""
        params = BatteryParams()

        result = validate_optimizer_energy_balance(
            df_results=perfect_balance_results, params=params, dt_hours=0.25, tolerance=1e-6
        )

        assert result["is_valid"] is True
        assert len(result["errors"]) == 0
        assert result["max_error"] < 1e-6
        assert result["summary"]["failed_steps"] == 0

    def test_validation_imperfect_balance(self, imperfect_balance_results: pd.DataFrame) -> None:
        """Test validation with energy balance errors."""
        params = BatteryParams()

        result = validate_optimizer_energy_balance(
            df_results=imperfect_balance_results,
            params=params,
            dt_hours=0.25,
            tolerance=1e-3,  # Larger tolerance to catch the intentional errors
        )

        assert result["is_valid"] is False
        assert len(result["errors"]) > 0
        assert result["max_error"] > 1e-3
        assert result["summary"]["failed_steps"] > 0

    def test_validation_custom_tolerance(self, imperfect_balance_results: pd.DataFrame) -> None:
        """Test validation with different tolerance levels."""
        params = BatteryParams()

        # Strict tolerance - should fail
        strict_result = validate_optimizer_energy_balance(
            df_results=imperfect_balance_results, params=params, dt_hours=0.25, tolerance=1e-6
        )

        # Loose tolerance - might pass
        loose_result = validate_optimizer_energy_balance(
            df_results=imperfect_balance_results,
            params=params,
            dt_hours=0.25,
            tolerance=100.0,  # Very large tolerance
        )

        # Strict should have more failures than loose
        assert strict_result["summary"]["failed_steps"] >= loose_result["summary"]["failed_steps"]

    def test_validation_single_step(self) -> None:
        """Test validation with single timestep (edge case)."""
        params = BatteryParams()
        single_step_df = pd.DataFrame(
            {"P_grid_MW": [40.0], "P_sol_bat_MW": [20.0], "P_dis_MW": [0.0], "E_MWh": [180.0]}
        )

        result = validate_optimizer_energy_balance(
            df_results=single_step_df, params=params, dt_hours=0.25
        )

        # Should be valid since no previous state to compare
        assert result["is_valid"] is True
        assert result["summary"]["validated_steps"] == 0


class TestExportOptimizerResults:
    """Test export_optimizer_results function."""

    @pytest.fixture
    def sample_results_for_export(self) -> pd.DataFrame:
        """Create sample results for export testing."""
        data = {
            "timestamp": ["2026-02-24T10:00:00", "2026-02-24T10:15:00"],
            "price_intraday": [50.0, 60.0],
            "solar_MW": [80.0, 90.0],
            "P_grid_MW": [40.0, 0.0],
            "P_dis_MW": [0.0, 30.0],
            "P_sol_bat_MW": [30.0, 50.0],
            "P_sol_sell_MW": [50.0, 40.0],
            "E_MWh": [180.0, 200.0],
        }
        return pd.DataFrame(data)

    def test_export_success(self, sample_results_for_export: pd.DataFrame) -> None:
        """Test successful export to CSV."""
        params = BatteryParams(p_rated_mw=100.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            export_optimizer_results(
                df_results=sample_results_for_export,
                csv_path=csv_path,
                params=params,
                dt_hours=0.25,
            )

            # Verify file was created
            assert Path(csv_path).exists()

            # Verify content by reading back
            import pandas as pd

            df_read = pd.read_csv(csv_path)
            assert len(df_read) == 2  # Two timesteps
            assert "simulation_step" in df_read.columns
            assert "energy_after_mwh" in df_read.columns

        finally:
            if Path(csv_path).exists():
                os.unlink(csv_path)


class TestCreateEnhancedOptimizerOutput:
    """Test create_enhanced_optimizer_output function."""

    @pytest.fixture
    def comprehensive_results(self) -> pd.DataFrame:
        """Create comprehensive optimizer results for testing."""
        data = {
            "timestamp": ["2026-02-24T10:00:00", "2026-02-24T10:15:00", "2026-02-24T10:30:00"],
            "price_intraday": [50.0, 60.0, 45.0],
            "solar_MW": [80.0, 90.0, 70.0],
            "P_grid_MW": [40.0, 0.0, 0.0],
            "P_dis_MW": [0.0, 0.0, 30.0],
            "P_sol_bat_MW": [30.0, 50.0, 0.0],
            "P_sol_sell_MW": [50.0, 40.0, 70.0],
            "E_MWh": [180.0, 200.0, 175.0],
            "z_grid": [1, 0, 0],
            "z_solbat": [0, 1, 0],
            "z_dis": [0, 0, 1],
            "cycle": [0.1, 0.2, 0.3],
        }
        return pd.DataFrame(data)

    def test_create_enhanced_output_success(self, comprehensive_results: pd.DataFrame) -> None:
        """Test successful creation of enhanced output."""
        params = BatteryParams(p_rated_mw=100.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            result = create_enhanced_optimizer_output(
                df_results=comprehensive_results,
                csv_path=csv_path,
                params=params,
                dt_hours=0.25,
                validate=True,
            )

            # Check return values
            assert result["csv_path"] == csv_path
            assert result["num_steps"] == 3
            assert "validation" in result
            assert "total_profit" in result
            assert "energy_range_mwh" in result

            # Verify CSV was created
            assert Path(csv_path).exists()
            csv_size = Path(csv_path).stat().st_size
            assert csv_size > 0

        finally:
            if Path(csv_path).exists():
                os.unlink(csv_path)

    def test_create_enhanced_output_no_validation(
        self, comprehensive_results: pd.DataFrame
    ) -> None:
        """Test enhanced output creation without validation."""
        params = BatteryParams(p_rated_mw=100.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            result = create_enhanced_optimizer_output(
                df_results=comprehensive_results,
                csv_path=csv_path,
                params=params,
                dt_hours=0.25,
                validate=False,  # No validation
            )

            # Should not include validation results
            assert "validation" not in result
            assert result["num_steps"] == 3

        finally:
            if Path(csv_path).exists():
                os.unlink(csv_path)

    def test_create_enhanced_output_with_config_path(
        self, comprehensive_results: pd.DataFrame
    ) -> None:
        """Test enhanced output with custom config path."""
        # Create temporary config file
        config_data = {
            "battery_params": {
                "p_rated_mw": 200.0,
                "eta_ch": 0.95,
                "eta_dis": 0.95,
                "a_aux": 0.01,
                "r_sd_per_hour": 0.001,
                "e_duration_hours": 4.0,
                "e_min_frac": 0.05,
                "e_max_frac": 0.95,
            },
            "simulation_defaults": {"dt_hours": 0.25, "enforce_bounds": True},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as config_f:
            json.dump(config_data, config_f)
            config_path = config_f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_f:
            csv_path = csv_f.name

        try:
            result = create_enhanced_optimizer_output(
                df_results=comprehensive_results,
                csv_path=csv_path,
                params=None,  # Should load from config
                config_path=config_path,
                validate=False,
            )

            assert result["num_steps"] == 3
            # Config should have been loaded (p_rated_mw=200.0)

        finally:
            for path in [config_path, csv_path]:
                if Path(path).exists():
                    os.unlink(path)


class TestWriteStepByStepBatteryLog:
    """Test write_step_by_step_battery_log function."""

    @pytest.fixture
    def sample_optimizer_row(self) -> pd.Series:
        """Create sample optimizer result row."""
        data = {
            "timestamp": "2026-02-24T10:00:00",
            "price_intraday": 50.0,
            "solar_MW": 80.0,
            "P_grid_MW": 40.0,
            "P_dis_MW": 0.0,
            "P_sol_bat_MW": 30.0,
            "P_sol_sell_MW": 50.0,
            "E_MWh": 180.0,
            "z_grid": 1,
            "z_solbat": 0,
            "z_dis": 0,
            "cycle": 0.1,
        }
        return pd.Series(data)

    def test_write_single_step_new_file(self, sample_optimizer_row: pd.Series) -> None:
        """Test writing single step to new file."""
        params = BatteryParams(p_rated_mw=100.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            log_entry = write_step_by_step_battery_log(
                optimizer_step=0,
                row=sample_optimizer_row,
                params=params,
                csv_path=csv_path,
                dt_hours=0.25,
                append_mode=False,  # Create new file
            )

            # Check returned log entry
            assert log_entry["simulation_step"] == 0
            assert log_entry["grid_power_mw"] == 40.0
            assert log_entry["solar_power_mw"] == 30.0
            assert log_entry["price_intraday"] == 50.0
            assert log_entry["battery_mode"] == "Grid→Battery"  # z_grid = 1

            # Verify file was created
            assert Path(csv_path).exists()

        finally:
            if Path(csv_path).exists():
                os.unlink(csv_path)

    def test_write_single_step_append_mode(self, sample_optimizer_row: pd.Series) -> None:
        """Test writing single step in append mode."""
        params = BatteryParams(p_rated_mw=100.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            # Write first step (new file)
            write_step_by_step_battery_log(
                optimizer_step=0,
                row=sample_optimizer_row,
                params=params,
                csv_path=csv_path,
                append_mode=False,
            )

            # Modify row for second step
            second_row = sample_optimizer_row.copy()
            second_row["P_grid_MW"] = 0.0
            second_row["z_grid"] = 0
            second_row["z_dis"] = 1
            second_row["E_MWh"] = 175.0

            # Write second step (append mode)
            log_entry = write_step_by_step_battery_log(
                optimizer_step=1, row=second_row, params=params, csv_path=csv_path, append_mode=True
            )

            assert log_entry["simulation_step"] == 1
            assert log_entry["battery_mode"] == "Battery→Grid"  # z_dis = 1

            # File should contain both steps
            df_read = pd.read_csv(csv_path)
            assert len(df_read) == 2

        finally:
            if Path(csv_path).exists():
                os.unlink(csv_path)

    def test_write_step_with_datetime(self, sample_optimizer_row: pd.Series) -> None:
        """Test writing step with explicit datetime."""
        params = BatteryParams(p_rated_mw=100.0)
        step_time = datetime(2026, 2, 24, 10, 30, 0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            log_entry = write_step_by_step_battery_log(
                optimizer_step=5,
                row=sample_optimizer_row,
                params=params,
                csv_path=csv_path,
                step_datetime=step_time,
                append_mode=False,
            )

            assert log_entry["timestamp_iso"] == "2026-02-24T10:30:00"
            assert log_entry["simulation_step"] == 5

        finally:
            if Path(csv_path).exists():
                os.unlink(csv_path)

    def test_battery_mode_detection(self, sample_optimizer_row: pd.Series) -> None:
        """Test battery mode detection from binary variables."""
        params = BatteryParams(p_rated_mw=100.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            # Test different modes
            test_cases = [
                ({"z_grid": 1, "z_solbat": 0, "z_dis": 0}, "Grid→Battery"),
                ({"z_grid": 0, "z_solbat": 1, "z_dis": 0}, "Solar→Battery"),
                ({"z_grid": 0, "z_solbat": 0, "z_dis": 1}, "Battery→Grid"),
                ({"z_grid": 0, "z_solbat": 0, "z_dis": 0}, "Idle"),
            ]

            for binary_vars, expected_mode in test_cases:
                test_row = sample_optimizer_row.copy()
                test_row.update(binary_vars)

                log_entry = write_step_by_step_battery_log(
                    optimizer_step=0,
                    row=test_row,
                    params=params,
                    csv_path=csv_path,
                    append_mode=False,
                )

                assert log_entry["battery_mode"] == expected_mode

        finally:
            if Path(csv_path).exists():
                os.unlink(csv_path)


class TestIntegration:
    """Integration tests for battery_inference functions."""

    def test_full_workflow_integration(self) -> None:
        """Test complete workflow from config loading to CSV export."""
        # Create sample optimizer results
        optimizer_results = pd.DataFrame(
            {
                "timestamp": ["2026-02-24T10:00:00", "2026-02-24T10:15:00", "2026-02-24T10:30:00"],
                "price_intraday": [50.0, 60.0, 45.0],
                "solar_MW": [80.0, 90.0, 70.0],
                "P_grid_MW": [40.0, 0.0, 0.0],
                "P_dis_MW": [0.0, 0.0, 30.0],
                "P_sol_bat_MW": [30.0, 50.0, 0.0],
                "P_sol_sell_MW": [50.0, 40.0, 70.0],
                "E_MWh": [180.0, 200.0, 175.0],
            }
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as csv_f:
            csv_path = csv_f.name

        try:
            # 1. Build params from BatteryConfiguration (replaces load_optimizer_battery_config)
            battery_config = BatteryConfiguration(
                rated_power_mw=100.0,
                energy_capacity_mwh=300.0,  # 100 MW * 3 h = 300 MWh
                min_soc_percent=10.0,
                max_soc_percent=90.0,
                charge_efficiency=0.97,
                discharge_efficiency=0.97,
                auxiliary_power_mw=0.5,
                self_discharge_rate_per_hour=0.0005,
            )
            params = battery_spec_to_params(battery_config)
            assert params.p_rated_mw == 100.0

            # 2. Create logs
            logs = create_optimizer_log_entries(
                df_results=optimizer_results, params=params, dt_hours=0.25
            )
            assert len(logs) == 3

            # 3. Validate energy balance
            validation = validate_optimizer_energy_balance(
                df_results=optimizer_results, params=params, dt_hours=0.25
            )
            # Note: May not be perfectly valid since we created artificial data
            assert "is_valid" in validation
            assert "summary" in validation

            # 4. Export enhanced output
            result = create_enhanced_optimizer_output(
                df_results=optimizer_results, csv_path=csv_path, params=params, validate=True
            )

            assert result["num_steps"] == 3
            assert Path(csv_path).exists()
            assert result["csv_path"] == csv_path

            # 5. Verify CSV content
            df_read = pd.read_csv(csv_path)
            assert len(df_read) == 3
            assert "simulation_step" in df_read.columns
            assert "energy_after_mwh" in df_read.columns
            assert "loss_total_mwh" in df_read.columns

        finally:
            if Path(csv_path).exists():
                os.unlink(csv_path)


if __name__ == "__main__":
    # Run all tests with detailed output
    pytest.main([__file__, "-v", "--tb=short"])
