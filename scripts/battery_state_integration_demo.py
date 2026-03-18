#!/usr/bin/env python3
"""
Demonstration of Battery State Integration with Optimizer

This script shows how to integrate real-time battery state with the optimizer,
replacing the previous hardcoded 50% SOC approach with actual battery telemetry.

Key improvements demonstrated:
1. Using BatteryTelemetry to capture current battery state
2. Processing telemetry with battery_status service to get validated BatteryState
3. Passing current state to optimizer instead of hardcoded values
4. Creating battery logs that accurately reflect the starting state

Example scenarios:
- Scenario A: Battery at 20% SOC, idle mode (low initial state)
- Scenario B: Battery at 80% SOC, charging mode (high initial state)
- Scenario C: Battery at 60% SOC, discharging, 2 cycles already used today
"""

import os
import sys
from datetime import datetime

# Add parent directory to path for module imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# Import the new battery domain models and services
from src.ors.domain.models.battery import BatterySpec, BatteryState, BatteryTelemetry
from src.ors.services.battery_status import update_battery_state

# Import optimizer with new interface
from src.ors.services.optimizer.optimizer import build_model


def create_dummy_telemetry_scenarios():
    """Create realistic battery telemetry scenarios for demonstration."""
    timestamp = datetime(2026, 3, 7, 8, 0, 0)  # 8 AM start time

    scenarios = {
        "low_soc_idle": BatteryTelemetry(
            timestamp=timestamp,
            current_energy_mwh=120.0,  # 20% of 600 MWh capacity
            current_soc_percent=20.0,
            current_power_mw=0.0,
            operating_mode="idle",
            is_available=True,
        ),
        "high_soc_charging": BatteryTelemetry(
            timestamp=timestamp,
            current_energy_mwh=480.0,  # 80% of 600 MWh capacity
            current_soc_percent=80.0,
            current_power_mw=75.0,  # Currently charging at 75 MW
            operating_mode="charging",
            is_available=True,
        ),
        "mid_soc_discharging": BatteryTelemetry(
            timestamp=timestamp,
            current_energy_mwh=360.0,  # 60% of 600 MWh capacity
            current_soc_percent=60.0,
            current_power_mw=-85.0,  # Currently discharging at 85 MW
            operating_mode="discharging",
            is_available=True,
        ),
        "missing_telemetry": BatteryTelemetry(
            timestamp=timestamp,
            current_energy_mwh=None,  # Missing energy data
            current_soc_percent=None,  # Missing SOC data
            current_power_mw=None,  # Missing power data
            operating_mode=None,  # Missing mode data
            is_available=True,
        ),
    }

    return scenarios


def create_battery_spec():
    """Create battery specification matching the optimizer configuration."""
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


def demonstrate_battery_state_processing(
    spec: BatterySpec, telemetry: BatteryTelemetry, scenario_name: str
):
    """Process raw telemetry into validated battery state."""
    print(f"\nInfo: Processing {scenario_name}:")
    print(
        f"   Raw telemetry: SOC={telemetry.current_soc_percent}%, Power={telemetry.current_power_mw}MW, Mode={telemetry.operating_mode}"
    )

    # Process telemetry with battery status service
    battery_state = update_battery_state(spec, telemetry)

    print(
        f"   Processed state: SOC={battery_state.soc_percent:.1f}%, Energy={battery_state.energy_mwh:.1f}MWh"
    )
    print(
        f"   Operating mode: {battery_state.operating_mode}, Available: {battery_state.is_available}"
    )

    if battery_state.quality_flags:
        print(f"Warning: Quality flags: {', '.join(battery_state.quality_flags)}")

    if battery_state.estimated_values:
        print(f"Info: Estimated values: {', '.join(battery_state.estimated_values)}")

    return battery_state


def run_optimizer_with_battery_state(
    battery_state: BatteryState, scenario_name: str, cycles_used: int = 0
):
    """Run optimizer using real battery state instead of hardcoded values."""
    print(f"\nInfo: Running optimizer for {scenario_name}:")

    # Load price and solar data (using dummy data for demonstration)
    # In real usage, this would come from price inference service
    price_data = {t: 50.0 + (t - 1) * 2.0 for t in range(1, 97)}  # Ascending price trend
    solar_data = {t: 10.0 if 20 <= t <= 80 else 0.0 for t in range(1, 97)}  # Solar during day
    p_30 = 65.0  # 30-day average price

    # Extract optimizer inputs from battery state
    initial_energy_mwh = battery_state.energy_mwh
    initial_mode = battery_state.operating_mode
    cycles_used_today = cycles_used

    print(
        f"   Initial energy: {initial_energy_mwh:.1f} MWh ({initial_energy_mwh/600.0*100:.1f}% SOC)"
    )
    print(f"   Initial mode: {initial_mode}")
    print(f"   Cycles used today: {cycles_used_today}")

    # Build and solve optimization model with real battery state
    model = build_model(
        price=price_data,
        solar=solar_data,
        p_30=p_30,
        initial_energy_mwh=initial_energy_mwh,
        initial_mode=initial_mode,
        cycles_used_today=cycles_used_today,
    )

    print(f"Success: Model built successfully with {len(price_data)} timesteps")
    print("Success: Initial state properly integrated (no more hardcoded 50% SOC)")

    return model


def main():
    """Demonstrate battery state integration with optimizer."""
    print("BATTERY STATE INTEGRATION DEMONSTRATION")
    print("=" * 60)

    # Create battery specification
    spec = create_battery_spec()
    print(f"Battery Specification: {spec.rated_power_mw}MW / {spec.energy_capacity_mwh}MWh")
    print(f"   SOC range: {spec.min_soc_percent}% - {spec.max_soc_percent}%")
    print(f"   Efficiency: charge={spec.charge_efficiency}, discharge={spec.discharge_efficiency}")

    # Create telemetry scenarios
    scenarios = create_dummy_telemetry_scenarios()

    # Process each scenario
    for scenario_name, telemetry in scenarios.items():
        # Process raw telemetry
        battery_state = demonstrate_battery_state_processing(spec, telemetry, scenario_name)

        # Simulate different cycle usage for different scenarios
        cycles_used = {"mid_soc_discharging": 2}.get(scenario_name, 0)

        # Run optimizer with real state
        run_optimizer_with_battery_state(battery_state, scenario_name, cycles_used)

        print("Info: Optimization ready - state properly initialized from real telemetry")

    print("\n" + "=" * 60)
    print("Success: Demonstration Complete!")
    print("\nIntegration Summary:")
    print("   ✓ BatteryTelemetry captures real-time sensor data")
    print("   ✓ battery_status service validates and processes telemetry")
    print("   ✓ Optimizer accepts actual battery state (no more hardcoded 50%)")
    print("   ✓ Battery logs start from actual state instead of assumptions")
    print("   ✓ Cycle counting accounts for already-used cycles")
    print("   ✓ Quality flags track data issues and estimations")
    print("Success: This solves the hardcoded battery status issue!")

    # Show before/after comparison
    print("\nBefore vs After Comparison:")
    print("   Before: E0 = E_CAP * 0.5  # Always 50% SOC hardcoded")
    print("   After:  E0 = battery_state.energy_mwh  # Real measured state")
    print("\n   Before: initial_mode = 'idle'  # Always started idle")
    print("   After:  initial_mode = battery_state.operating_mode  # Actual mode")
    print("\n   Before: cycles_used = 0  # Never tracked daily usage")
    print("   After:  cycles_used = counted_from_daily_logs()  # Real cycle count")


if __name__ == "__main__":
    main()
