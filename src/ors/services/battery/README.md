# Battery Management Module

A comprehensive Python module for battery energy storage system (BESS) simulation, analysis, and data management. This module provides core battery physics calculations, configuration management, and CSV data export capabilities.

## Overview

The `battery_management.py` module implements the battery energy update equation:

```
E_t = E_{t-1} + eta_ch * P_ch * dt - (P_dis * dt) / eta_dis - P_aux * dt - E_{t-1} * r_sd * dt
```

Where:
- `P_ch` = Grid power + Solar power (charging)
- `dt` is the time step in hours
- Power values are in MW
- Energy values are in MWh

## Quick Start

```python
from battery_management import (
    BatteryParams, 
    step_energy, 
    compute_losses,
    load_battery_params, 
    write_simulation_csv
)

# Load battery configuration
params = load_battery_params('battery_config.json')

# Simulate one time step
new_energy = step_energy(
    e_prev_mwh=150.0,
    p_grid_mw=50.0,
    p_sol_mw=30.0,
    p_dis_mw=0.0,
    params=params,
    dt_hours=0.25
)
```

## Core Components

### Classes

#### `BatteryParams`
Battery parameter container with physical and operational constraints.

**Components:**
- `battery_management.py`: Core battery physics and simulation
- `battery_status.py`: Battery telemetry processing and state validation  
- `demo.py`: Example usage and simulation patterns
- `test.py`: Comprehensive unit tests

#### `BatteryParams`
Battery parameter container with physical and operational constraints.

**Properties:**
- `p_rated_mw` (float): Rated power in MW
- `eta_ch` (float): Charging efficiency (0-1)
- `eta_dis` (float): Discharging efficiency (0-1) 
- `a_aux` (float): Auxiliary power fraction of rated power
- `r_sd_per_hour` (float): Self-discharge rate per hour
- `e_duration_hours` (float): Energy capacity in hours at rated power
- `e_min_frac` (float): Minimum energy fraction (0-1)
- `e_max_frac` (float): Maximum energy fraction (0-1)

**Validation:**
All parameters are validated for physical consistency:
- `p_rated_mw` > 0
- `eta_ch`, `eta_dis` in (0, 1] (0% < efficiency ≤ 100%)
- `a_aux`, `r_sd_per_hour` ≥ 0
- `e_duration_hours` > 0
- `e_min_frac`, `e_max_frac` in [0, 1]
- `e_min_frac` ≤ `e_max_frac`

Invalid values raise `ValueError` with descriptive messages.

**Computed Properties:**
- `p_aux_mw`: Auxiliary power draw (MW)
- `e_cap_mwh`: Total energy capacity (MWh)
- `e_min_mwh`: Minimum energy threshold (MWh)
- `e_max_mwh`: Maximum energy threshold (MWh)

**Example:**
```python
params = BatteryParams(
    p_rated_mw=100.0,
    eta_ch=0.97,
    eta_dis=0.97,
    e_duration_hours=4.0
)
print(f"Battery capacity: {params.e_cap_mwh} MWh")
```

#### `LossBreakdown`
Container for detailed energy loss components in a simulation step.

**Properties:**
- `loss_charge_ineff_mwh`: Energy lost due to charging inefficiency
- `loss_discharge_ineff_mwh`: Energy lost due to discharging inefficiency
- `loss_aux_mwh`: Energy consumed by auxiliary systems
- `loss_self_discharge_mwh`: Energy lost to self-discharge
- `total_loss_mwh`: Total losses (computed property)

**Example:**
```python
losses = compute_losses(
    e_prev_mwh=150.0,
    p_grid_mw=50.0,
    p_sol_mw=0.0,
    p_dis_mw=0.0,
    params=params,
    dt_hours=1.0
)
print(f"Total losses: {losses.total_loss_mwh} MWh")
```

### Core Physics Functions

#### `step_energy()`
Updates battery energy state for one time step using the battery physics equation.

**Parameters:**
- `e_prev_mwh`: Previous energy state (MWh)
- `p_grid_mw`: Grid power (MW, positive for charging)
- `p_sol_mw`: Solar power (MW, positive for charging) 
- `p_dis_mw`: Discharge power (MW, positive for discharging)
- `params`: BatteryParams object
- `dt_hours`: Time step duration (hours)
- `enforce_bounds`: Whether to clamp energy to min/max limits

**Returns:** New energy state (MWh)

**Example:**
```python
energy_next = step_energy(
    e_prev_mwh=150.0,
    p_grid_mw=40.0,    # Charging from grid
    p_sol_mw=20.0,     # Charging from solar
    p_dis_mw=0.0,      # No discharge
    params=params,
    dt_hours=0.25,     # 15-minute step
    enforce_bounds=True
)
```

#### `compute_losses()`
Calculates detailed energy loss breakdown for one time step.

**Parameters:**
- `e_prev_mwh`: Previous energy state (MWh)
- `p_grid_mw`: Grid power (MW)
- `p_sol_mw`: Solar power (MW)
- `p_dis_mw`: Discharge power (MW)
- `params`: BatteryParams object
- `dt_hours`: Time step duration (hours)

**Returns:** LossBreakdown object with detailed loss components

**Example:**
```python
losses = compute_losses(
    e_prev_mwh=200.0,
    p_grid_mw=0.0,
    p_sol_mw=0.0,
    p_dis_mw=50.0,     # Discharging 50 MW
    params=params,
    dt_hours=1.0
)
print(f"Discharge inefficiency: {losses.loss_discharge_ineff_mwh} MWh")
```

#### `clamp()`
Utility function to constrain a value within specified bounds.

**Parameters:**
- `value`: Value to constrain
- `lo`: Lower bound
- `hi`: Upper bound

**Returns:** Constrained value

### Configuration Management

#### `load_config()`
Loads configuration from JSON file with validation.

**Parameters:**
- `config_path`: Path to JSON configuration file

**Returns:** Dictionary containing configuration data

**Raises:**
- `FileNotFoundError`: If config file doesn't exist
- `ValueError`: If JSON is invalid or missing required sections

#### `load_battery_params()`
Loads BatteryParams from JSON configuration file.

**Parameters:**
- `config_path`: Path to JSON configuration file

**Returns:** BatteryParams object

**Example:**
```python
params = load_battery_params('battery_config.json')
```

#### `load_simulation_defaults()`
Loads simulation default settings from JSON configuration.

**Parameters:**
- `config_path`: Path to JSON configuration file

**Returns:** Dictionary containing simulation defaults

#### `load_battery_params_and_defaults()`
Convenience function to load both battery parameters and simulation defaults.

**Parameters:**
- `config_path`: Path to JSON configuration file

**Returns:** Tuple of (BatteryParams, simulation_defaults)

**Example:**
```python
params, defaults = load_battery_params_and_defaults('battery_config.json')
simulator = BatterySimulator(
    params=params, 
    dt_hours=defaults['dt_hours'],
    enforce_bounds=defaults['enforce_bounds']
)
```

### CSV Export Functions

#### `write_simulation_csv()`
Exports simulation logs to CSV file with standardized column headers.

**Parameters:**
- `logs`: List of log entry dictionaries
- `csv_path`: Output CSV file path (default: "battery_storage.csv")

**Raises:**
- `ValueError`: If logs list is empty
- `IOError`: If unable to write to file

**CSV Columns:**
- `simulation_step`: Step number (0-based)
- `elapsed_time_hours`: Cumulative simulation time
- `time_step_hours`: Duration of this step
- `grid_power_mw`: Grid charging/discharging power
- `solar_power_mw`: Solar charging power
- `discharge_power_mw`: Battery discharge power
- `energy_before_mwh`: Energy before step
- `energy_after_mwh`: Energy after step
- `energy_in_from_grid_mwh`: Energy input from grid
- `energy_in_from_solar_mwh`: Energy input from solar
- `energy_in_total_mwh`: Total energy input
- `energy_out_total_mwh`: Total energy output
- `loss_charging_inefficiency_mwh`: Charging losses
- `loss_discharge_inefficiency_mwh`: Discharge losses
- `loss_auxiliary_power_mwh`: Auxiliary power losses
- `loss_self_discharge_mwh`: Self-discharge losses
- `loss_total_mwh`: Total losses
- `timestamp_iso`: ISO timestamp (optional)

#### `create_log_entry()`
Creates standardized log entry dictionary for CSV export.

**Parameters:**
- `step`: Step number (starting from 0)
- `t_hours`: Elapsed time in hours
- `dt_hours`: Time step duration in hours
- `p_grid_mw`: Grid power in MW
- `p_sol_mw`: Solar power in MW
- `p_dis_mw`: Discharge power in MW
- `e_prev_mwh`: Energy before step in MWh
- `e_next_mwh`: Energy after step in MWh
- `losses`: LossBreakdown object
- `eta_ch`: Charging efficiency
- `timestamp_iso`: Optional ISO timestamp string

**Returns:** Dictionary ready for CSV export

**Example:**
```python
log_entry = create_log_entry(
    step=0,
    t_hours=0.0,
    dt_hours=0.25,
    p_grid_mw=50.0,
    p_sol_mw=20.0,
    p_dis_mw=0.0,
    e_prev_mwh=150.0,
    e_next_mwh=165.5,
    losses=losses,
    eta_ch=params.eta_ch
)
```

## Configuration File Format

The module expects JSON configuration files with the following structure:

```json
{
  "battery_params": {
    "p_rated_mw": 100.0,
    "eta_ch": 0.97,
    "eta_dis": 0.97,
    "a_aux": 0.005,
    "r_sd_per_hour": 0.0005,
    "e_duration_hours": 3.0,
    "e_min_frac": 0.1,
    "e_max_frac": 0.9
  },
  "simulation_defaults": {
    "dt_hours": 0.25,
    "enforce_bounds": true
  }
}
```

## Usage Examples

### Basic Energy Step Calculation
```python
from battery_management import BatteryParams, step_energy

# Create battery with 100 MW, 4-hour duration
params = BatteryParams(p_rated_mw=100.0, e_duration_hours=4.0)

# Simulate charging for 15 minutes
energy_new = step_energy(
    e_prev_mwh=200.0,  # Start at 50% SOC
    p_grid_mw=75.0,    # Charge at 75 MW
    p_sol_mw=25.0,     # Additional 25 MW from solar
    p_dis_mw=0.0,      # No discharge
    params=params,
    dt_hours=0.25      # 15-minute step
)
print(f"Energy after charging: {energy_new:.1f} MWh")
```

### Full Simulation with CSV Export
```python
from battery_management import *
from datetime import datetime

# Load configuration
params, defaults = load_battery_params_and_defaults('battery_config.json')

# Initialize simulation
logs = []
energy = 150.0  # Starting energy

# Run simulation for 24 steps (6 hours at 15-min steps)
for step in range(24):
    # Define power profiles (example)
    p_grid = 50.0 if step < 12 else -25.0  # Charge then discharge
    p_solar = 30.0 if 4 <= step <= 16 else 0.0  # Solar during day
    p_discharge = 60.0 if step >= 16 else 0.0  # Evening discharge
    
    # Calculate losses
    losses = compute_losses(
        e_prev_mwh=energy,
        p_grid_mw=p_grid,
        p_sol_mw=p_solar,
        p_dis_mw=p_discharge,
        params=params,
        dt_hours=defaults['dt_hours']
    )
    
    # Update energy
    energy_new = step_energy(
        e_prev_mwh=energy,
        p_grid_mw=p_grid,
        p_sol_mw=p_solar,
        p_dis_mw=p_discharge,
        params=params,
        dt_hours=defaults['dt_hours']
    )
    
    # Create log entry
    log_entry = create_log_entry(
        step=step,
        t_hours=step * defaults['dt_hours'],
        dt_hours=defaults['dt_hours'],
        p_grid_mw=p_grid,
        p_sol_mw=p_solar,
        p_dis_mw=p_discharge,
        e_prev_mwh=energy,
        e_next_mwh=energy_new,
        losses=losses,
        eta_ch=params.eta_ch,
        timestamp_iso=datetime.now().isoformat()
    )
    
    logs.append(log_entry)
    energy = energy_new

# Export to CSV
write_simulation_csv(logs, 'simulation_results.csv')
print(f"Simulation complete. Final energy: {energy:.1f} MWh")
```

### Loss Analysis
```python
# Analyze charging losses
charging_losses = compute_losses(
    e_prev_mwh=150.0,
    p_grid_mw=80.0,   # High charging rate
    p_sol_mw=20.0,
    p_dis_mw=0.0,
    params=params,
    dt_hours=1.0
)

print("Charging Loss Analysis:")
print(f"  Charging inefficiency: {charging_losses.loss_charge_ineff_mwh:.3f} MWh")
print(f"  Auxiliary power: {charging_losses.loss_aux_mwh:.3f} MWh")
print(f"  Self-discharge: {charging_losses.loss_self_discharge_mwh:.3f} MWh")
print(f"  Total losses: {charging_losses.total_loss_mwh:.3f} MWh")

# Analyze discharging losses
discharging_losses = compute_losses(
    e_prev_mwh=300.0,
    p_grid_mw=0.0,
    p_sol_mw=0.0,
    p_dis_mw=80.0,   # High discharge rate
    params=params,
    dt_hours=1.0
)

print("\nDischarging Loss Analysis:")
print(f"  Discharge inefficiency: {discharging_losses.loss_discharge_ineff_mwh:.3f} MWh")
print(f"  Auxiliary power: {discharging_losses.loss_aux_mwh:.3f} MWh")
print(f"  Self-discharge: {discharging_losses.loss_self_discharge_mwh:.3f} MWh")
print(f"  Total losses: {discharging_losses.total_loss_mwh:.3f} MWh")
```

## Testing

The module includes comprehensive unit tests covering all functions and classes:

```bash
# Run all tests
python test_comprehensive.py

# Run with pytest (if installed)
python -m pytest test_comprehensive.py -v
```

Test coverage includes:
- Parameter validation and edge cases
- Energy equation accuracy
- Loss calculation correctness  
- Configuration loading and error handling
- CSV export functionality
- Immutability of data classes

## Demo Usage

See `demo.py` for a complete example of building battery simulations using this module. The demo shows:
- Configuration loading
- Custom simulation loop construction
- Realistic power profile generation
- Result analysis and CSV export
- Performance metrics calculation

```bash
python demo.py
```

---

## Module: `battery_status.py` — Battery State Processing

Processes raw battery telemetry into validated state objects with comprehensive quality tracking and error handling.

### Purpose

The `battery_status` module transforms real-time battery sensor data into reliable state information suitable for optimization and control decisions. It handles missing data, validates consistency between different measurements, and enforces physical constraints.

### Key Functions

#### `update_battery_state(spec, telemetry, *, prev_state=None, power_threshold_mw=0.1)`
Main function for processing battery telemetry into validated state.

**Parameters:**
- `spec`: BatterySpec defining system constraints and capabilities
- `telemetry`: BatteryTelemetry with raw sensor measurements
- `prev_state`: Optional previous state for consistency validation
- `power_threshold_mw`: Power threshold for determining idle mode (default: 0.1 MW)

**Processing Logic:**
1. **Energy/SOC Resolution**: Prioritizes energy over SOC, estimates missing values
2. **Cross-Validation**: Checks consistency between energy and SOC measurements
3. **Constraint Enforcement**: Clamps energy and power within specification limits
4. **Operating Mode Determination**: Derives mode from power measurements and telemetry
5. **Quality Assessment**: Tracks missing data, estimates, and validation issues

**Quality Flags Generated:**
- `missing_energy_and_soc`: Both primary measurements unavailable
- `energy_estimated_from_soc`: Energy calculated from SOC percentage  
- `energy_soc_mismatch`: Energy and SOC measurements inconsistent (>1% difference)
- `using_previous_state`: Fallback to previous state for missing data
- `defaulted_to_50_percent_soc`: Safe default when no state information available
- `energy_clamped_to_minimum/maximum`: Energy constrained to operational bounds
- `missing_power_data`: Power measurement unavailable
- `power_clamped_to_rated`: Power exceeded rated limits
- `mode_power_mismatch`: Operating mode inconsistent with power measurement
- `mode_estimated_from_power`: Operating mode derived from power flow
- `unrealistic_energy_change`: Energy change exceeds physical limits
- `invalid_mode_defaulted_to_idle`: Invalid mode corrected to safe default

#### `estimate_energy_from_soc(soc_percent, energy_capacity_mwh)`
Converts state of charge percentage to energy using battery capacity.

#### `estimate_soc_from_energy(energy_mwh, energy_capacity_mwh)` 
Converts stored energy to state of charge percentage.

#### `determine_operating_mode(power_mw, power_threshold_mw=0.1)`
Determines battery operating mode from power flow measurement.

**Returns:** `"charging"`, `"discharging"`, or `"idle"` based on power direction and threshold

### Integration with Battery Management

The `battery_status` module works alongside `battery_management` to provide complete battery system modeling:

- **battery_management.py**: Physics-based simulation and energy calculations
- **battery_status.py**: Real-time data processing and state validation

Together, they enable both forward simulation (physics) and state estimation (telemetry processing) capabilities.

### Example Usage

```python
from ors.domain.models.battery import BatterySpec, BatteryTelemetry
from ors.services.battery.battery_status import update_battery_state
from datetime import datetime

# Define battery system
spec = BatterySpec(
    rated_power_mw=100.0,
    energy_capacity_mwh=400.0,
    min_soc_percent=10.0,
    max_soc_percent=90.0
)

# Process telemetry with missing data
telemetry = BatteryTelemetry(
    timestamp=datetime(2026, 3, 11, 14, 30, 0),
    current_energy_mwh=None,     # Missing
    current_soc_percent=65.0,    # Available
    current_power_mw=50.0,       # Charging
    operating_mode=None,         # Missing
    is_available=True
)

# Generate validated state
state = update_battery_state(spec, telemetry)

# Results with quality tracking
print(f"Energy: {state.energy_mwh} MWh")  # Estimated from SOC
print(f"Mode: {state.operating_mode}")    # Estimated from power
print(f"Quality flags: {state.quality_flags}")
# {'energy_estimated_from_soc', 'mode_estimated_from_power'}
```

---

## Dependencies

- **Python 3.9+**: Type hints and `|` union syntax
- **Standard Library Only**: `json`, `csv`, `pathlib`, `datetime`, `typing`
- **Optional**: `pytest` for running tests

## Design Philosophy

- **Separation of Concerns**: Core physics separate from simulation logic
- **Immutability**: Data classes prevent accidental modification
- **Type Safety**: Full type hints for better IDE support
- **Comprehensive Testing**: 100% function coverage with edge cases
- **Clear Documentation**: Self-documenting code with detailed docstrings
- **Standard Library**: No external dependencies for core functionality

This module focuses on providing robust, well-tested battery physics calculations while leaving simulation design and optimization to the user.