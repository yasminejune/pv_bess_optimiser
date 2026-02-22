#!/usr/bin/env python3
"""Demo script showing how to build battery simulations.

Demonstrates:
1. Loading configuration from JSON
2. Building a custom simulation using battery_management functions
3. Collecting detailed logging data
4. Exporting results to CSV
5. Basic analysis of simulation results

This shows the pattern for building simulations that use battery_management
functions rather than having battery_management run simulations itself.
"""

from datetime import datetime, timedelta
from typing import Any

# Import handling for both mypy (relative) and direct execution (absolute)
try:
    # When run from project root with mypy or as a module
    from .battery_management import (
        BatteryParams,
        compute_losses,
        create_log_entry,
        load_battery_params_and_defaults,
        step_energy,
        write_simulation_csv,
    )
except (ImportError, ModuleNotFoundError):
    # When run directly from this directory
    from battery_management import (  # type: ignore[import-not-found,no-redef]
        BatteryParams,
        compute_losses,
        create_log_entry,
        load_battery_params_and_defaults,
        step_energy,
        write_simulation_csv,
    )


class BatterySimulator:
    """Battery simulation manager that uses battery_management functions.

    This demonstrates how to build simulations that use the battery management
    module without having battery_management run the simulation itself.
    """

    def __init__(self, params: BatteryParams, dt_hours: float, enforce_bounds: bool = True) -> None:
        """Initialize simulator with battery configuration.

        Args:
            params: Battery parameters
            dt_hours: Time step duration in hours
            enforce_bounds: Whether to enforce energy bounds
        """
        self.params = params
        self.dt_hours = dt_hours
        self.enforce_bounds = enforce_bounds
        self.logs: list[dict[str, Any]] = []

    def run_simulation(
        self,
        initial_energy_mwh: float,
        power_profiles: dict[str, list[float]],
        start_datetime: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Run battery simulation with detailed logging.

        Args:
            initial_energy_mwh: Starting battery energy in MWh
            power_profiles: Dict with keys 'grid', 'solar', 'discharge'
                           and values as lists of power values in MW
            start_datetime: Optional start time for timestamping

        Returns:
            List of detailed log entries for each simulation step
        """
        # Validate input data
        p_grid = power_profiles["grid"]
        p_solar = power_profiles["solar"]
        p_discharge = power_profiles["discharge"]

        if not (len(p_grid) == len(p_solar) == len(p_discharge)):
            raise ValueError("All power profile arrays must have the same length")

        # Initialize simulation state
        energy_current = float(initial_energy_mwh)
        self.logs = []

        # Run simulation step by step
        for step in range(len(p_grid)):
            # Extract power values for this step
            p_grid_step = p_grid[step]
            p_solar_step = p_solar[step]
            p_discharge_step = p_discharge[step]

            # Compute losses using battery_management function
            losses = compute_losses(
                e_prev_mwh=energy_current,
                p_grid_mw=p_grid_step,
                p_sol_mw=p_solar_step,
                p_dis_mw=p_discharge_step,
                params=self.params,
                dt_hours=self.dt_hours,
            )

            # Update energy state using battery_management function
            energy_next = step_energy(
                e_prev_mwh=energy_current,
                p_grid_mw=p_grid_step,
                p_sol_mw=p_solar_step,
                p_dis_mw=p_discharge_step,
                params=self.params,
                dt_hours=self.dt_hours,
                enforce_bounds=self.enforce_bounds,
            )

            # Calculate timing
            t_hours = step * self.dt_hours
            timestamp_iso = None
            if start_datetime is not None:
                step_datetime = start_datetime + timedelta(hours=t_hours)
                timestamp_iso = step_datetime.isoformat()

            # Create log entry using csv_logger utility
            log_entry = create_log_entry(
                step=step,
                t_hours=t_hours,
                dt_hours=self.dt_hours,
                p_grid_mw=p_grid_step,
                p_sol_mw=p_solar_step,
                p_dis_mw=p_discharge_step,
                e_prev_mwh=energy_current,
                e_next_mwh=energy_next,
                losses=losses,
                eta_ch=self.params.eta_ch,
                timestamp_iso=timestamp_iso,
            )

            self.logs.append(log_entry)

            # Update energy state for next step
            energy_current = energy_next

        return self.logs

    def export_to_csv(self, csv_path: str) -> None:
        """Export simulation results to CSV file.

        Args:
            csv_path: Path where CSV should be written
        """
        if not self.logs:
            raise ValueError("No simulation data to export. Run simulation first.")

        write_simulation_csv(self.logs, csv_path)


def create_example_power_profiles(num_steps: int) -> dict[str, list[float]]:
    """Create example power profiles for demonstration.

    This simulates a pattern like:
    - Morning: grid charging + solar ramp-up
    - Midday: peak solar + reduced grid
    - Afternoon: solar decline + grid export
    - Evening: discharge for grid services
    - Night: minimal activity

    Args:
        num_steps: Number of time steps to generate

    Returns:
        Dictionary with 'grid', 'solar', 'discharge' power profiles in MW
    """
    grid_power = []
    solar_power = []
    discharge_power = []

    for i in range(num_steps):
        # Create daily patterns (assuming 15-minute steps)
        hour_of_day = (i * 0.25) % 24

        # Grid power: charge in morning, export in afternoon
        if 6 <= hour_of_day <= 10:  # Morning charging
            grid_mw = 40.0 - (hour_of_day - 6) * 5  # Decline as solar increases
        elif 14 <= hour_of_day <= 17:  # Afternoon export
            grid_mw = -20.0  # Export to grid
        elif 22 <= hour_of_day <= 24 or 0 <= hour_of_day <= 5:  # Night
            grid_mw = 10.0  # Light charging
        else:
            grid_mw = 0.0  # Idle

        # Solar power: bell curve during day
        if 6 <= hour_of_day <= 18:
            # Peak at noon, zero at 6am and 6pm
            solar_angle = (hour_of_day - 12) / 6  # -1 to +1
            solar_mw = 50.0 * max(0, 1 - solar_angle**2)  # Bell curve
        else:
            solar_mw = 0.0

        # Discharge power: peak in evening
        if 17 <= hour_of_day <= 21:  # Evening peak
            discharge_mw = 60.0  # High discharge for grid services
        elif 11 <= hour_of_day <= 13:  # Midday peak shaving
            discharge_mw = 30.0
        else:
            discharge_mw = 0.0

        grid_power.append(grid_mw)
        solar_power.append(solar_mw)
        discharge_power.append(discharge_mw)

    return {"grid": grid_power, "solar": solar_power, "discharge": discharge_power}


def analyze_simulation_results(logs: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze simulation results and return summary statistics.

    Args:
        logs: Simulation log entries

    Returns:
        Dictionary containing analysis results
    """
    if not logs:
        return {}

    # Energy analysis
    initial_energy = logs[0]["energy_before_mwh"]
    final_energy = logs[-1]["energy_after_mwh"]
    energy_values = [initial_energy] + [log["energy_after_mwh"] for log in logs]

    # Power and energy flows
    total_grid_energy = sum(log["energy_in_from_grid_mwh"] for log in logs)
    total_solar_energy = sum(log["energy_in_from_solar_mwh"] for log in logs)
    total_discharge_energy = sum(log["energy_out_total_mwh"] for log in logs)
    total_energy_in = total_grid_energy + total_solar_energy

    # Loss analysis
    total_losses = sum(log["loss_total_mwh"] for log in logs)
    loss_breakdown = {
        "charging_inefficiency": sum(log["loss_charging_inefficiency_mwh"] for log in logs),
        "discharge_inefficiency": sum(log["loss_discharge_inefficiency_mwh"] for log in logs),
        "auxiliary_power": sum(log["loss_auxiliary_power_mwh"] for log in logs),
        "self_discharge": sum(log["loss_self_discharge_mwh"] for log in logs),
    }

    # Efficiency calculation
    round_trip_efficiency = None
    if total_energy_in > 0 and total_discharge_energy > 0:
        round_trip_efficiency = (total_discharge_energy / total_energy_in) * 100

    return {
        "energy": {
            "initial_mwh": initial_energy,
            "final_mwh": final_energy,
            "change_mwh": final_energy - initial_energy,
            "min_mwh": min(energy_values),
            "max_mwh": max(energy_values),
        },
        "flows": {
            "grid_energy_in_mwh": total_grid_energy,
            "solar_energy_in_mwh": total_solar_energy,
            "total_energy_in_mwh": total_energy_in,
            "discharge_energy_out_mwh": total_discharge_energy,
        },
        "losses": {"total_mwh": total_losses, "breakdown": loss_breakdown},
        "efficiency": {"round_trip_percent": round_trip_efficiency},
        "simulation": {
            "num_steps": len(logs),
            "duration_hours": logs[-1]["elapsed_time_hours"] + logs[-1]["time_step_hours"],
        },
    }


def main() -> None:
    """Run demonstration simulation and analysis."""
    print("Battery Simulation Demo")
    print("=" * 40)

    # 1. Load configuration
    config_path = "src/ors/services/battery/battery_config.json"

    try:
        params, defaults = load_battery_params_and_defaults(config_path)
        print(f"✓ Loaded configuration from {config_path}")
        print(f"  Battery: {params.p_rated_mw} MW / {params.e_cap_mwh} MWh")
        print(f"  Time step: {defaults['dt_hours']} hours")
    except FileNotFoundError:
        print(f"⚠ Config file {config_path} not found. Using default parameters.")
        params = BatteryParams()
        defaults = {"dt_hours": 0.25, "enforce_bounds": True}

    # 2. Create simulator instance
    simulator = BatterySimulator(
        params=params, dt_hours=defaults["dt_hours"], enforce_bounds=defaults["enforce_bounds"]
    )

    # 3. Generate power profiles for 24 hours (96 steps at 15-min intervals)
    print("\n📊 Setting up simulation...")

    num_steps = 96  # 24 hours at 15-minute intervals
    power_profiles = create_example_power_profiles(num_steps)
    initial_energy = 150.0  # Start at 50% SOC for 300 MWh battery
    start_time = datetime(2024, 6, 15, 0, 0, 0)  # Midnight start

    print(f"  Initial energy: {initial_energy} MWh")
    print(f"  Simulation steps: {num_steps}")
    print(f"  Total duration: {num_steps * defaults['dt_hours']} hours")
    print(f"  Time step: {defaults['dt_hours']} hours ({defaults['dt_hours']*60} minutes)")

    # 4. Run simulation
    print("\n🔄 Running simulation...")

    try:
        logs = simulator.run_simulation(
            initial_energy_mwh=initial_energy,
            power_profiles=power_profiles,
            start_datetime=start_time,
        )
        print(f"✓ Simulation completed: {len(logs)} steps processed")

        # Export to CSV
        simulator.export_to_csv("src/ors/services/battery/battery_storage.csv")
        print("✓ Results exported to src/ors/services/battery/battery_storage.csv")

    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        return

    # 5. Analyze results
    print("\n📈 Analysis Results:")
    print("-" * 30)

    analysis = analyze_simulation_results(logs)

    # Energy analysis
    energy = analysis["energy"]
    print(f"Energy Range: {energy['min_mwh']:.1f} - {energy['max_mwh']:.1f} MWh")
    print(f"Final Energy: {energy['final_mwh']:.1f} MWh")
    print(f"Net Change: {energy['change_mwh']:+.1f} MWh")

    # Energy flows
    flows = analysis["flows"]
    print("\nEnergy Flows:")
    print(f"  Grid energy in: {flows['grid_energy_in_mwh']:.2f} MWh")
    print(f"  Solar energy in: {flows['solar_energy_in_mwh']:.2f} MWh")
    print(f"  Total energy in: {flows['total_energy_in_mwh']:.2f} MWh")
    print(f"  Total energy out: {flows['discharge_energy_out_mwh']:.2f} MWh")

    # Loss breakdown
    losses = analysis["losses"]
    total_losses = losses["total_mwh"]
    print("\nLoss Breakdown:")
    for loss_type, loss_value in losses["breakdown"].items():
        percentage = (loss_value / total_losses) * 100 if total_losses > 0 else 0
        readable_name = loss_type.replace("_", " ").title()
        print(f"  {readable_name}: {loss_value:.3f} MWh ({percentage:.1f}%)")
    print(f"  Total losses: {total_losses:.3f} MWh")

    # Efficiency
    efficiency = analysis["efficiency"]["round_trip_percent"]
    if efficiency is not None:
        print(f"\nRound-trip efficiency: {efficiency:.1f}%")

    # Peak examples
    print("\n📊 Peak Activity Examples:")
    print("Time  | Grid | Solar | Disc | Energy | Losses")
    print("-" * 50)

    # Show a few representative time steps
    sample_indices = [24, 48, 72]  # 6am, noon, 6pm
    for i in sample_indices:
        if i < len(logs):
            log = logs[i]
            hour = int(log["elapsed_time_hours"]) % 24
            print(
                f"{hour:02d}:00 | "
                f"{log['grid_power_mw']:5.1f} | "
                f"{log['solar_power_mw']:5.1f} | "
                f"{log['discharge_power_mw']:4.1f} | "
                f"{log['energy_after_mwh']:6.1f} | "
                f"{log['loss_total_mwh']:6.3f}"
            )

    print("\n✅ Demo completed successfully!")
    print("\n📊 This demonstrates how to build simulations using:")
    print("  - config_loader.py: Load battery configuration")
    print("  - battery_management.py: Core battery physics (step_energy, compute_losses)")
    print("  - csv_logger.py: Export simulation data")
    print("\n🔧 Check src/ors/services/battery/battery_storage.csv for full simulation data.")
    print("📝 Modify power_profiles in this script to test different scenarios.")


if __name__ == "__main__":
    main()
