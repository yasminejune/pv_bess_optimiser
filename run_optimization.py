"""Main orchestration file for running optimizations from JSON configuration.

This file provides the main entry point for running optimizations using
client-provided JSON configuration files. It handles:
- Configuration loading and validation
- Data loading from various sources
- Battery state initialization
- Optimization execution
- Results processing and output generation

To run do python run_optimization.py config_templates/FILE_NAME.json
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pyomo.environ import SolverFactory, value

# Add project root to Python path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import src.ors.services.optimizer.optimizer as opt_module
from src.ors.config.optimization_config import OptimizationConfig, load_config_from_json
from src.ors.domain.models.battery import BatterySpec, BatteryState, BatteryTelemetry
from src.ors.services.battery.battery_management import BatteryParams
from src.ors.services.battery.battery_status import update_battery_state
from src.ors.services.battery_to_optimization.battery_inference import (
    create_enhanced_optimizer_output,
)


class OptimizationRunner:
    """Orchestrates complete optimization runs from configuration."""

    def __init__(self, config: OptimizationConfig) -> None:
        """Initialize optimization runner with configuration.

        Args:
            config: Validated optimization configuration
        """
        self.config = config
        self.results: dict[str, Any] = {}
        self.verbose = config.output.verbose

    def run(self) -> dict[str, Any]:
        """Execute complete optimization run.

        Returns:
            Dictionary with run results and metadata

        Raises:
            Exception: If optimization fails at any stage
        """
        try:
            self._print_header()

            # Stage 1: Load and prepare data
            self._print_stage("1. Loading Data")
            price_data, terminal_price, solar_data = self._load_data()

            # Stage 2: Initialize battery state
            self._print_stage("2. Initializing Battery State")
            battery_state, battery_spec = self._initialize_battery_state()

            # Stage 3: Build and solve optimization model
            self._print_stage("3. Running Optimization")
            model_results = self._run_optimization(
                price_data, solar_data, terminal_price, battery_state, battery_spec
            )

            # Stage 4: Process and save results
            self._print_stage("4. Processing Results")
            output_results = self._process_results(model_results, battery_spec)

            # Stage 5: Generate summary and recommendations
            self._print_stage("5. Generating Summary")
            summary = self._generate_summary(output_results)

            # Compile final results
            self.results = {
                "config": self.config,
                "data_summary": {
                    "price_range": f"£{min(price_data.values()):.2f} - £{max(price_data.values()):.2f}/MWh",
                    "solar_total": (
                        f"{sum(solar_data.values()):.1f} MWh" if self.config.has_pv else "No PV"
                    ),
                    "terminal_price": f"£{terminal_price:.2f}/MWh",
                },
                "battery_state": {
                    "initial_soc": f"{battery_state.soc_percent:.1f}%",
                    "initial_mode": battery_state.operating_mode,
                    "available": battery_state.is_available,
                },
                "optimization_results": output_results,
                "summary": summary,
                "run_timestamp": datetime.now(),
                "success": True,
            }

            self._print_completion()
            return self.results

        except Exception as e:
            self._print_error(f"Optimization failed: {e}")
            self.results = {
                "config": self.config,
                "error": str(e),
                "success": False,
                "run_timestamp": datetime.now(),
            }
            raise

    def _load_data(self) -> tuple[dict[int, float], float, dict[int, float]]:
        """Load price and solar data using direct service calls like backtesting does."""
        start_datetime = self.config.optimization_start_datetime
        end_datetime = self.config.optimization_end_datetime
        time_step_minutes = self.config.optimization.time_step_minutes

        if self.verbose:
            print("Info: Loading data using direct service calls...")

        # Load price data - use dummy data like backtesting does for now
        price_data, terminal_price = self._create_dummy_price_data(
            start_datetime, end_datetime, time_step_minutes
        )

        # Load solar data using the same service backtesting uses
        if self.config.has_pv:
            try:
                from datetime import timezone

                from src.ors.config.pv_config import SiteType, get_pv_config
                from src.ors.services.weather_to_pv import generate_pv_power_for_date_range

                if self.verbose:
                    print("Info: Generating PV power forecast...")

                # Use same config as backtesting
                pv_config = get_pv_config(SiteType.BURST_1)

                # Convert timezone-naive datetimes to timezone-aware UTC datetimes
                start_datetime_utc = start_datetime.replace(tzinfo=timezone.utc)
                end_datetime_utc = end_datetime.replace(tzinfo=timezone.utc)

                # Generate PV data
                df = generate_pv_power_for_date_range(
                    config=pv_config,
                    start_datetime=start_datetime_utc,
                    end_datetime=end_datetime_utc,
                )

                # Convert to MW dict (kW -> MW conversion like backtesting)
                df["generation_MW"] = df["generation_kw"] / 1000.0
                solar_data = {
                    i + 1: float(df.iloc[i]["generation_MW"])
                    for i in range(min(len(df), self.config.total_time_steps))
                }

                # Fill any missing timesteps with zero
                for i in range(len(solar_data) + 1, self.config.total_time_steps + 1):
                    solar_data[i] = 0.0

            except Exception as e:
                print(f"Warning: Solar generation failed: {e}")
                print("Info: Using dummy solar pattern")
                solar_data = self._create_dummy_solar_data(
                    start_datetime, end_datetime, time_step_minutes
                )
        else:
            solar_data = dict.fromkeys(range(1, self.config.total_time_steps + 1), 0.0)

        # Validate data consistency
        if len(price_data) != len(solar_data):
            # Pad shorter dict with zeros
            target_len = self.config.total_time_steps
            if len(price_data) < target_len:
                avg_price = sum(price_data.values()) / len(price_data)
                for i in range(len(price_data) + 1, target_len + 1):
                    price_data[i] = avg_price
            if len(solar_data) < target_len:
                for i in range(len(solar_data) + 1, target_len + 1):
                    solar_data[i] = 0.0

        self._print_data_summary(price_data, solar_data, terminal_price)

        return price_data, terminal_price, solar_data

    def _initialize_battery_state(self) -> tuple[BatteryState, BatterySpec]:
        """Initialize battery state from configuration."""
        # Create battery specification from config
        battery_config = self.config.battery
        battery_spec = BatterySpec(
            rated_power_mw=battery_config.rated_power_mw,
            energy_capacity_mwh=battery_config.energy_capacity_mwh,
            min_soc_percent=battery_config.min_soc_percent,
            max_soc_percent=battery_config.max_soc_percent,
            charge_efficiency=battery_config.charge_efficiency,
            discharge_efficiency=battery_config.discharge_efficiency,
            auxiliary_power_mw=battery_config.auxiliary_power_mw,
            self_discharge_rate_per_hour=battery_config.self_discharge_rate_per_hour,
        )

        # Create battery telemetry from current state
        telemetry = BatteryTelemetry(
            timestamp=self.config.optimization_start_datetime,
            current_energy_mwh=battery_config.current_energy_mwh,
            current_soc_percent=battery_config.current_soc_percent,
            current_power_mw=battery_config.current_power_mw,
            operating_mode=battery_config.current_mode,
            is_available=battery_config.is_available,
        )

        # Process telemetry to get validated battery state
        battery_state = update_battery_state(battery_spec, telemetry)

        if self.verbose:
            print(
                f"Info: Battery: {battery_spec.rated_power_mw}MW / {battery_spec.energy_capacity_mwh}MWh"
            )
            print(
                f"Info: Current state: {battery_state.soc_percent:.1f}% SOC, {battery_state.operating_mode} mode"
            )
            if battery_state.quality_flags:
                print(f"Warning: Quality flags: {', '.join(battery_state.quality_flags)}")

        return battery_state, battery_spec

    def _run_optimization(
        self,
        price_data: dict[int, float],
        solar_data: dict[int, float],
        terminal_price: float,
        battery_state: BatteryState,
        battery_spec: BatterySpec,
    ) -> dict[str, Any]:
        """Build and solve optimization model using backtesting pattern."""
        # Extract initial state
        initial_energy_mwh = battery_state.energy_mwh
        cycles_used_today = self.config.battery.cycles_used_today

        if self.verbose:
            print(f"Info: Building model with {len(price_data)} timesteps...")
            print(
                f"Info: Initial: {initial_energy_mwh:.1f} MWh ({battery_state.soc_percent:.1f}% SOC)"
            )
            print(f"Info: Cycles used: {cycles_used_today}")

        # Set module-level E0 variable like backtesting does
        original_e0 = opt_module.E0
        opt_module.E0 = initial_energy_mwh

        try:
            # Build optimization model with backtesting signature
            model = opt_module.build_model(
                price_data,
                solar_data,
                terminal_price,
                cycles_used_today,
                len(price_data),  # t_boundary
                q_init=0,
                z_dis_init=0,
                verbose=self.verbose,
                battery_spec=battery_spec,
                time_step_hours=self.config.optimization.time_step_minutes / 60.0,
            )

            # Solve model
            if self.verbose:
                print("Info: Solving optimization model...")

            # Try multiple solvers like backtesting does
            solver = None
            for solver_name in ["highs", "glpk", "cbc", "gurobi", "cplex"]:
                try:
                    test_solver = SolverFactory(solver_name)
                    if test_solver.available():
                        solver = test_solver
                        if self.verbose:
                            print(f"Info: Using solver: {solver_name}")
                        break
                except Exception:
                    continue

            if solver is None:
                raise RuntimeError(
                    "No optimization solver available. Install one of: highs, glpk, cbc, gurobi, cplex"
                )

            # Solve with minimal output
            result = solver.solve(model, tee=self.verbose)

            # Check solution status
            from pyomo.opt import SolverStatus, TerminationCondition

            if (
                result.solver.status != SolverStatus.ok
                or result.solver.termination_condition
                not in {
                    TerminationCondition.optimal,
                    TerminationCondition.feasible,
                }
            ):
                raise RuntimeError(
                    f"Optimization failed: status={result.solver.status}, "
                    f"termination={result.solver.termination_condition}"
                )

            # Extract results using value() like backtesting does
            model_results = self._extract_model_results(model, price_data, solar_data)

            if self.verbose:
                total_profit = model_results.get("total_profit", 0)
                print(f"Success: Optimization completed - Total profit: £{total_profit:.2f}")

            return model_results

        finally:
            # Restore original E0 like backtesting does
            opt_module.E0 = original_e0

    def _process_results(
        self, model_results: dict[str, Any], battery_spec: BatterySpec
    ) -> dict[str, Any]:
        """Process optimization results and create output files."""
        # Create results DataFrame
        results_df = pd.DataFrame(model_results["timestep_results"])
        results_df["timestamp"] = pd.date_range(
            start=self.config.optimization_start_datetime,
            periods=len(results_df),
            freq=f"{self.config.optimization.time_step_minutes}min",
        )

        # Save main results CSV
        output_path = Path(self.config.output.output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"Info: Saving results to {output_path}...")

        results_df.to_csv(output_path, index=False)

        # Create enhanced battery logs if requested
        detailed_logs = None
        if self.config.output.detailed_log_path:
            log_path = Path(self.config.output.detailed_log_path)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            if self.verbose:
                print(f"Info: Creating detailed battery logs: {log_path}...")

            try:
                # Convert BatterySpec to BatteryParams for validation consistency
                battery_params = self._convert_battery_spec_to_params(battery_spec)

                # Use the battery integration module
                detailed_logs = create_enhanced_optimizer_output(
                    df_results=results_df,
                    csv_path=str(log_path),
                    params=battery_params,
                    dt_hours=self.config.optimization.time_step_minutes / 60,
                    start_datetime=self.config.optimization_start_datetime,
                    validate=True,
                    initial_energy_mwh=model_results["initial_energy_mwh"],
                    verbose=self.verbose,
                )

                if self.verbose and detailed_logs:
                    print(
                        f"Success: Detailed logs created with {detailed_logs['num_steps']} entries"
                    )

            except Exception as e:
                print(f"Warning: Detailed logs failed: {e}")
                detailed_logs = None

        return {
            "main_csv_path": str(output_path),
            "detailed_log_path": str(log_path) if self.config.output.detailed_log_path else None,
            "results_df": results_df,
            "detailed_logs": detailed_logs,
            "model_results": model_results,
        }

    def _generate_summary(self, output_results: dict[str, Any]) -> dict[str, Any]:
        """Generate optimization summary and recommendations."""
        results_df = output_results["results_df"]
        model_results = output_results["model_results"]

        # Time period summary
        start_datetime_str = self.config.optimization_start_datetime.strftime("%Y-%m-%d %H:%M")
        end_datetime_str = self.config.optimization_end_datetime.strftime("%Y-%m-%d %H:%M")

        # Financial summary
        total_profit = model_results["total_profit"]
        total_cycles = model_results["total_cycles"]

        # Energy summary
        final_energy_mwh = model_results["final_energy_mwh"]
        final_soc_percent = (final_energy_mwh / self.config.battery.energy_capacity_mwh) * 100

        energy_stats = {
            "initial_energy": model_results["initial_energy_mwh"],
            "final_energy": final_energy_mwh,
            "final_soc_percent": final_soc_percent,
            "max_energy": results_df["E_MWh"].max(),
            "min_energy": results_df["E_MWh"].min(),
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(results_df, model_results)

        summary = {
            "period": f"{start_datetime_str} to {end_datetime_str}",
            "duration_hours": self.config.optimization.duration_hours,
            "financial": {
                "total_profit": total_profit,
                "profit_per_hour": total_profit / self.config.optimization.duration_hours,
                "currency": self.config.output.currency,
            },
            "energy": energy_stats,
            "operations": {
                "cycles_used": total_cycles,
                "max_cycles_per_day": self.config.battery.max_cycles_per_day,
                "cycle_efficiency": total_cycles / self.config.battery.max_cycles_per_day * 100,
            },
            "recommendations": recommendations,
        }

        return summary

    def _generate_recommendations(
        self, results_df: pd.DataFrame, model_results: dict[str, Any]
    ) -> list[str]:
        """Generate actionable recommendations from results."""
        recommendations = []

        # Analyze operating patterns
        charging_periods = results_df[results_df["z_grid"] + results_df["z_solbat"] > 0.5]
        discharging_periods = results_df[results_df["z_dis"] > 0.5]

        if len(charging_periods) > 0:
            avg_charge_price = (
                charging_periods["price_intraday"] * charging_periods["P_grid_MW"]
            ).sum() / charging_periods["P_grid_MW"].sum()
            recommendations.append(f"Average charging price: £{avg_charge_price:.2f}/MWh")

        if len(discharging_periods) > 0:
            avg_discharge_price = (
                discharging_periods["price_intraday"] * discharging_periods["P_dis_MW"]
            ).sum() / discharging_periods["P_dis_MW"].sum()
            recommendations.append(f"Average discharge price: £{avg_discharge_price:.2f}/MWh")

        # Cycle efficiency
        cycles_used = model_results["total_cycles"]
        max_cycles = self.config.battery.max_cycles_per_day
        if cycles_used < max_cycles:
            recommendations.append(
                f"Could utilize {max_cycles - cycles_used} additional cycles for more arbitrage opportunities"
            )

        # Solar integration
        if self.config.has_pv:
            solar_to_battery = results_df["P_sol_bat_MW"].sum()
            solar_direct = results_df["P_sol_sell_MW"].sum()
            if solar_to_battery > 0:
                recommendations.append(
                    f"Stored {solar_to_battery:.1f} MWh of solar generation for later discharge"
                )
            if solar_direct > 0:
                recommendations.append(
                    f"Directly exported {solar_direct:.1f} MWh of solar generation"
                )

        return recommendations

    def _extract_model_results(
        self, model: Any, price_data: dict[int, float], solar_data: dict[int, float]
    ) -> dict[str, Any]:
        """Extract results from solved optimization model."""
        timestep_results = []
        total_profit = 0.0
        total_cycles = 0

        for t in model.T:
            step_result = {
                "timestep": t,
                "price_intraday": price_data[t],
                "solar_MW": solar_data[t],
                "P_grid_MW": value(model.P_grid[t]),
                "P_dis_MW": value(model.P_dis[t]),
                "P_sol_bat_MW": value(model.P_sol_bat[t]),
                "P_sol_sell_MW": value(model.P_sol_sell[t]),
                "E_MWh": value(model.E[t]),
                "z_grid": int(round(value(model.z_grid[t]))),
                "z_solbat": int(round(value(model.z_solbat[t]))),
                "z_dis": int(round(value(model.z_dis[t]))),
                "q_flag": int(round(value(model.q[t]))),
                "s_dis": int(round(value(model.s_dis[t]))),
                "cycle": int(round(value(model.cycle[t]))),
            }

            # Calculate step profit
            dt = self.config.optimization.time_step_minutes / 60.0  # Convert to hours
            step_profit = (
                price_data[t]
                * (
                    step_result["P_dis_MW"]
                    + 0.97 * step_result["P_sol_sell_MW"]  # Solar sell efficiency
                    - step_result["P_grid_MW"]
                )
                * dt
            )
            step_result["profit_step"] = step_profit

            total_profit += step_profit
            total_cycles += step_result["cycle"]

            timestep_results.append(step_result)

        return {
            "timestep_results": timestep_results,
            "total_profit": total_profit,
            "total_cycles": total_cycles,
            "initial_energy_mwh": value(model.E[1]) if hasattr(model, "E") else None,
            "final_energy_mwh": value(model.E[len(model.T)]) if hasattr(model, "E") else None,
            "objective_value": value(model.obj) if hasattr(model, "obj") else None,
        }

    def _create_fallback_prices(
        self, start_datetime: datetime, end_datetime: datetime, time_step_minutes: int
    ) -> tuple[dict[int, float], float]:
        """Create fallback price data when loading fails."""
        return self._create_dummy_price_data(start_datetime, end_datetime, time_step_minutes)

    def _create_fallback_solar(
        self, start_datetime: datetime, end_datetime: datetime, time_step_minutes: int
    ) -> dict[int, float]:
        """Create fallback solar data when loading fails."""
        return self._create_dummy_solar_data(start_datetime, end_datetime, time_step_minutes)

    def _create_dummy_price_data(
        self, start_datetime: datetime, end_datetime: datetime, time_step_minutes: int
    ) -> tuple[dict[int, float], float]:
        """Generate dummy price data with realistic daily patterns."""
        import math

        num_steps = int((end_datetime - start_datetime).total_seconds() / 60 / time_step_minutes)
        price_dict = {}

        # Generate realistic daily price pattern
        for i in range(1, num_steps + 1):
            hour_of_day = ((i - 1) * time_step_minutes / 60) % 24
            # Higher prices during peak hours (7-9am, 7-10pm)
            if 7 <= hour_of_day <= 9 or 19 <= hour_of_day <= 22:
                base_price = 60 + 20 * math.sin(hour_of_day / 24 * 2 * math.pi)
            else:
                base_price = 40 + 10 * math.sin(hour_of_day / 24 * 2 * math.pi)
            price_dict[i] = max(20, base_price)

        # Average price as terminal price
        terminal_price = sum(price_dict.values()) / len(price_dict)
        return price_dict, terminal_price

    def _create_dummy_solar_data(
        self, start_datetime: datetime, end_datetime: datetime, time_step_minutes: int
    ) -> dict[int, float]:
        """Generate dummy solar generation data."""
        import math

        num_steps = int((end_datetime - start_datetime).total_seconds() / 60 / time_step_minutes)
        solar_dict = {}

        # Get rated power from config or default
        rated_power_mw = self.config.pv.rated_power_kw / 1000.0 if self.config.pv else 1.0

        for i in range(1, num_steps + 1):
            hour_of_day = ((i - 1) * time_step_minutes / 60) % 24
            # Simple solar curve: peak at noon, zero at night
            if 6 <= hour_of_day <= 18:
                solar_fraction = math.sin((hour_of_day - 6) / 12 * math.pi) ** 2
                solar_dict[i] = rated_power_mw * solar_fraction
            else:
                solar_dict[i] = 0.0

        return solar_dict

    def _print_header(self) -> None:
        """Print optimization run header."""
        print("\n" + "=" * 80)
        print("Battery Energy Storage Optimization")
        print("=" * 80)
        print(f"Configuration: {self.config.config_name}")
        print(
            f"Period: {self.config.optimization.optimization_date} ({self.config.optimization.duration_hours}h)"
        )
        print(
            f"Battery: {self.config.battery.rated_power_mw} MW / {self.config.battery.energy_capacity_mwh} MWh"
        )
        if self.config.has_pv:
            print(f"Solar: {self.config.pv.rated_power_kw} kW PV system")
        print()

    def _print_stage(self, stage: str) -> None:
        """Print stage header."""
        print(f"\n{stage}")
        print("-" * len(stage))

    def _print_data_summary(
        self, price_data: dict[int, float], solar_data: dict[int, float], terminal_price: float
    ) -> None:
        """Print data loading summary."""
        print(
            f"Success: Loaded {len(price_data)} price points (£{min(price_data.values()):.2f} - £{max(price_data.values()):.2f} /MWh)"
        )
        print(
            f"Success: Loaded {len(solar_data)} solar points ({sum(solar_data.values()):.1f} MWh total)"
        )
        print(f"Success: Terminal price: £{terminal_price:.2f} /MWh")

    def _print_completion(self) -> None:
        """Print completion summary."""
        print("\n" + "=" * 80)
        print("Success: Optimization completed successfully")
        print("=" * 80)

        summary = self.results["summary"]
        print("\nResults Summary")
        print(f"Period: {summary['period']}")
        print(f"Total Profit: £{summary['financial']['total_profit']:.2f}")
        print(
            f"Cycles Used: {summary['operations']['cycles_used']}/{summary['operations']['max_cycles_per_day']}"
        )
        print(
            f"Battery charge at end of {summary['duration_hours']}-hour period: {summary['energy']['final_soc_percent']:.1f}%"
        )

        print("\nOutput Files:")
        print(f"  - Main results: {self.results['optimization_results']['main_csv_path']}")
        if self.results["optimization_results"]["detailed_log_path"]:
            print(f"  - Detailed logs: {self.results['optimization_results']['detailed_log_path']}")

        # Print specific recommendation for first timestep
        self._print_next_recommendation()
        print(
            f"\nInfo: All subsequent timesteps through the end of the optimization horizon are available in {self.results['optimization_results']['main_csv_path']}."
        )

    def _get_next_recommendation(self) -> dict[str, any] | None:
        """Get recommendation for first 15-minute timestep."""
        try:
            from datetime import timedelta

            # Calculate target timestamp (start + 15 minutes)
            target_time = self.config.optimization_start_datetime + timedelta(minutes=15)

            # Read the results CSV that was just written
            csv_path = self.results["optimization_results"]["main_csv_path"]
            df = pd.read_csv(csv_path)

            # Convert timestamp column to datetime if needed
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Find the closest matching row at or after target time
            future_rows = df[df["timestamp"] >= target_time]
            if len(future_rows) == 0:
                # Fallback to closest row if no future rows
                time_diffs = abs(df["timestamp"] - target_time)
                closest_idx = time_diffs.idxmin()
                target_row = df.iloc[closest_idx]
            else:
                target_row = future_rows.iloc[0]

            # Infer action from power flows
            action = self._infer_action(target_row)

            return {
                "timestamp": target_row["timestamp"],
                "action": action,
                "grid_power": target_row.get("P_grid_MW", 0),
                "solar_power": target_row.get("P_sol_bat_MW", 0),
                "solar_export": target_row.get("P_sol_sell_MW", 0),
                "discharge_power": target_row.get("P_dis_MW", 0),
                "energy_after": target_row.get("E_MWh", 0),
                "price": target_row.get("price_intraday", 0),
            }

        except Exception as e:
            if self.verbose:
                print(f"Warning: Could not generate specific recommendation: {e}")
            return None

    def _infer_action(self, row: dict) -> str:
        """Infer the main action from power flow values."""
        grid_power = row.get("P_grid_MW", 0)
        solar_bat = row.get("P_sol_bat_MW", 0)
        solar_export = row.get("P_sol_sell_MW", 0)
        discharge = row.get("P_dis_MW", 0)

        # Priority order for action classification
        if discharge > 0.1:
            return "Battery discharge"
        elif grid_power > 0.1:
            return "Grid to battery"
        elif solar_bat > 0.1:
            return "Solar to battery"
        elif solar_export > 0.1:
            return "Solar export"
        else:
            return "Idle"

    def _print_next_recommendation(self) -> None:
        """Print specific recommendation for first timestep."""
        recommendation = self._get_next_recommendation()

        if recommendation is None:
            print("\nInfo: See the CSV files for timestep-level recommendations.")
            return

        print("\nNext recommended step")
        print("-" * 21)

        # Format timestamp nicely
        timestamp = recommendation["timestamp"]
        if hasattr(timestamp, "strftime"):
            time_str = timestamp.strftime("%Y-%m-%d %H:%M")
        else:
            time_str = str(timestamp)

        print(f"Time: {time_str}")
        print(f"Action: {recommendation['action']}")
        print(f"Grid power: {recommendation['grid_power']:.2f} MW")

        # Only show solar power if PV system is configured
        if self.config.has_pv:
            print(f"Solar to battery: {recommendation['solar_power']:.2f} MW")
            print(f"Solar export: {recommendation['solar_export']:.2f} MW")

        print(f"Discharge power: {recommendation['discharge_power']:.2f} MW")
        print(f"Energy after step: {recommendation['energy_after']:.2f} MWh")
        print(f"Price: £{recommendation['price']:.2f}/MWh")

    def _convert_battery_spec_to_params(self, battery_spec: BatterySpec) -> BatteryParams:
        """Convert BatterySpec from main config to BatteryParams for validation.

        Args:
            battery_spec: Battery specification from main config

        Returns:
            BatteryParams: Equivalent battery parameters for validation
        """
        # Calculate duration from capacity and power rating
        e_duration_hours = battery_spec.energy_capacity_mwh / battery_spec.rated_power_mw

        # Calculate auxiliary power fraction from absolute value
        a_aux = battery_spec.auxiliary_power_mw / battery_spec.rated_power_mw

        # Convert SOC percentages to fractions
        e_min_frac = battery_spec.min_soc_percent / 100.0
        e_max_frac = battery_spec.max_soc_percent / 100.0

        return BatteryParams(
            p_rated_mw=battery_spec.rated_power_mw,
            eta_ch=battery_spec.charge_efficiency,
            eta_dis=battery_spec.discharge_efficiency,
            a_aux=a_aux,
            r_sd_per_hour=battery_spec.self_discharge_rate_per_hour,
            e_duration_hours=e_duration_hours,
            e_min_frac=e_min_frac,
            e_max_frac=e_max_frac,
        )

    def _print_error(self, error_msg: str) -> None:
        """Print error message."""
        print(f"\nError: {error_msg}")
        print("Info: Check your configuration file and try again.")


def main() -> None:
    """Main entry point for running optimizations."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run battery energy storage optimization from JSON configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s config.json                    # Run optimization with config file
  %(prog)s config.json --verbose          # Run with detailed logging
  %(prog)s --validate config.json        # Just validate configuration

Configuration files should be in JSON format. See config_templates/ for examples.
        """,
    )
    parser.add_argument("config_file", help="Path to JSON configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--validate", action="store_true", help="Only validate configuration (don't run)"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        print(f"Info: Loading configuration from {args.config_file}...")
        config = load_config_from_json(args.config_file)
        print("Success: Configuration loaded and validated successfully")

        if args.validate:
            print("Success: Configuration is valid!")
            print(
                f"  Period: {config.optimization.optimization_date} ({config.optimization.duration_hours}h)"
            )
            print(
                f"  Battery: {config.battery.rated_power_mw}MW / {config.battery.energy_capacity_mwh}MWh"
            )
            if config.has_pv:
                print(f"  PV: {config.pv.rated_power_kw}kW")
            return

        # Override verbose setting from command line
        if args.verbose:
            config.output.verbose = True

        # Run optimization
        runner = OptimizationRunner(config)
        runner.run()

        # Exit with success code
        sys.exit(0)

    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: Optimization failed: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
