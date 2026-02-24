# Battery-Optimizer Integration Module

## Overview

This module provides seamless integration between the optimization engine and the battery management system, enabling detailed step-by-step battery operation logging and validation. It bridges optimizer results with the battery module's physics calculations and CSV logging capabilities.

## Key Features

- **Step-by-step battery logging**: Records every 15-minute timestep with complete energy accounting
- **Battery physics validation**: Verifies optimizer energy states against battery management equations
- **Standardized CSV output**: Compatible with battery module analysis tools
- **Detailed loss breakdown**: Tracks charging inefficiency, discharge inefficiency, auxiliary power, and self-discharge
- **Energy flow tracking**: Complete accounting of energy in/out flows

## File Structure

```
battery_to_optimization/
├── battery_inference.py     # Core integration functions
├── battery_storage.csv      # Output: Step-by-step battery logs
└── README.md               # This file
```

## Dependencies

- Battery management module (`../battery/`)
- Pandas for data processing
- Battery configuration JSON file
- Optimizer results DataFrame

## Core Functions

### `load_optimizer_battery_config()`
Loads battery parameters from the battery module configuration.

**Returns:** 
- `BatteryParams`: Battery configuration object
- `dict`: Simulation defaults (time step, bounds, etc.)

### `create_optimizer_log_entries()`
Converts optimizer results into detailed battery step logs.

**Parameters:**
- `df_results`: Optimizer results DataFrame
- `params`: Battery parameters
- `dt_hours`: Time step duration (default: 0.25 hours)
- `start_datetime`: Optional timestamp for logs

**Returns:**
- `List[Dict]`: Detailed log entries for each timestep

### `create_enhanced_optimizer_output()`
Main function that processes optimizer results and creates battery CSV.

**Parameters:**
- `df_results`: Optimizer results DataFrame
- `csv_path`: Output CSV file path
- `params`: Battery parameters (optional, loaded from config if None)
- `dt_hours`: Time step duration
- `start_datetime`: Optional start time
- `validate`: Whether to validate energy balance

**Returns:**
- `Dict`: Processing results with validation info

## Input Requirements

### Optimizer Results DataFrame
The optimizer must provide a DataFrame with these columns:

**Required Columns:**
- `timestamp`: ISO timestamp for each step
- `price_intraday`: Energy price (€/MWh)
- `solar_MW`: Total solar generation (MW)
- `P_grid_MW`: Grid charging power (MW, positive = charging)
- `P_dis_MW`: Battery discharge power (MW)
- `P_sol_bat_MW`: Solar power to battery (MW)
- `P_sol_sell_MW`: Solar power sold directly to grid (MW)
- `E_MWh`: Battery energy state (MWh)

**Optional Binary Variables:**
- `z_grid`: Grid charging mode (0/1)
- `z_solbat`: Solar charging mode (0/1)
- `z_dis`: Discharge mode (0/1)
- `q_flag`: Charge flag for cycle counting
- `s_dis`: Start of discharge indicator
- `cycle`: Cycle event indicator

### Battery Configuration
Uses standard battery module configuration:
```json
{
  "battery_params": {
    "p_rated_mw": 100.0,
    "eta_ch": 0.97,
    "eta_dis": 0.97,
    "a_aux": 0.005,
    "r_sd_per_hour": 0.0005,
    "e_duration_hours": 3.0,
    "e_min_frac": 0.10,
    "e_max_frac": 0.90
  },
  "simulation_defaults": {
    "dt_hours": 0.25,
    "enforce_bounds": true
  }
}
```

## Output Format

### Battery Storage CSV
The output CSV contains detailed battery operation logs:

**Standard Battery Module Fields:**
- `simulation_step`: Step number (0-based)
- `elapsed_time_hours`: Cumulative time
- `timestamp_iso`: ISO timestamp
- `time_step_hours`: Step duration
- `grid_power_mw`: Grid charging power
- `solar_power_mw`: Solar charging power
- `discharge_power_mw`: Battery discharge power
- `energy_before_mwh`: Energy before step
- `energy_after_mwh`: Energy after step
- `energy_in_from_grid_mwh`: Grid energy input
- `energy_in_from_solar_mwh`: Solar energy input
- `energy_in_total_mwh`: Total energy input
- `energy_out_total_mwh`: Total energy output
- `loss_charging_inefficiency_mwh`: Charging losses
- `loss_discharge_inefficiency_mwh`: Discharge losses
- `loss_auxiliary_power_mwh`: Auxiliary power consumption
- `loss_self_discharge_mwh`: Self-discharge losses
- `loss_total_mwh`: Total losses

## Implementation Guide

### 1. Basic Integration

```python
from battery_inference import create_enhanced_optimizer_output

# After optimization completes
export_results = create_enhanced_optimizer_output(
    df_results=optimizer_output,
    csv_path="battery_storage.csv",
    validate=True
)
```

### 2. With Custom Configuration

```python
from battery_inference import (
    load_optimizer_battery_config,
    create_enhanced_optimizer_output
)

# Load specific battery configuration
params, defaults = load_optimizer_battery_config("custom_battery_config.json")

# Process results with custom parameters
export_results = create_enhanced_optimizer_output(
    df_results=optimizer_output,
    csv_path="battery_storage.csv",
    params=params,
    dt_hours=defaults["dt_hours"],
    start_datetime=datetime(2024, 6, 15, 0, 0, 0),
    validate=True
)
```

### 3. Integration in Optimizer Script

```python
# In optimizer.py
from battery_inference import create_enhanced_optimizer_output

# After solving optimization model...
out = process_optimizer_results()

# Create battery logs
try:
    start_datetime = datetime(2024, 6, 15, 0, 0, 0)
    
    export_results = create_enhanced_optimizer_output(
        df_results=out,
        csv_path="src/ors/services/battery_to_optimization/battery_storage.csv",
        params=battery_params,
        dt_hours=DT,
        start_datetime=start_datetime,
        validate=True
    )
    
    print(f"✓ Battery logs saved: {export_results['csv_path']}")
    print(f"  Steps processed: {export_results['num_steps']}")
    
    # Check validation results
    if 'validation' in export_results:
        validation = export_results['validation']
        if validation['is_valid']:
            print(f"✓ Energy balance validation: PASSED")
        else:
            print(f"⚠ Energy balance validation: FAILED")
            
except Exception as e:
    print(f"⚠ Battery logging failed: {e}")
```

## Validation Features

The module includes automatic validation of optimizer energy states against battery physics:

- **Energy balance verification**: Checks that optimizer energy calculations match battery module physics
- **Loss consistency**: Validates that energy flows account for all loss components
- **Bounds checking**: Ensures energy states remain within battery limits

### Validation Output

```python
validation_results = {
    'is_valid': True/False,
    'max_error': float,  # Maximum energy discrepancy (MWh)
    'errors': [list of error details],
    'summary': {
        'total_steps': int,
        'failed_steps': int,
        'max_error_mwh': float,
        'avg_error_mwh': float
    }
}
```

## Best Practices

1. **Always validate**: Set `validate=True` to catch energy balance issues early
2. **Use timestamps**: Provide `start_datetime` for proper time series analysis
3. **Check file output**: Verify CSV file creation and content
4. **Monitor validation**: Review validation results for optimization quality
5. **Consistent time steps**: Ensure optimizer and battery module use same `dt_hours`

## Troubleshooting

### Common Issues

**Empty CSV files**: 
- Check that optimizer DataFrame has required columns
- Verify battery configuration file exists and is valid
- Ensure all numeric values are finite (not NaN/inf)

**Validation failures**:
- Check optimizer energy balance equations match battery physics
- Verify power values are correctly signed (positive for charging)
- Ensure time step consistency between optimizer and battery module

**Import errors**:
- Verify battery module is in correct relative path (`../battery/`)
- Check that all required packages are installed
- Ensure Python path includes the battery module

## Examples

### Typical Output Log
```
📝 Creating detailed step-by-step battery logs for 96 timesteps...
   Processing step 0...
     Powers: Grid=0.00, Solar=0.00, Discharge=100.00
     Energy: 150.00 -> 150.00 MWh
     ✓ Step 0 logged
  Step   1 (00:00): Battery→Grid | E= 150.0MWh | Loss=0.917MWh
  
✓ Created 96 detailed step logs
✓ Successfully wrote 96 step records to CSV
✓ CSV file created: 12076 bytes
✓ Energy balance validation: PASSED
  Max error: 0.000000 MWh
```

### CSV Sample
```csv
simulation_step,elapsed_time_hours,timestamp_iso,grid_power_mw,solar_power_mw,discharge_power_mw,energy_before_mwh,energy_after_mwh,loss_total_mwh
0,0.0,2024-06-15T00:00:00,0.0,0.0,100.0,150.0,150.0,0.916946
1,0.25,2024-06-15T00:15:00,100.0,0.0,0.0,150.0,174.106,0.89375
```

This integration enables complete traceability of battery operations during optimization, providing detailed energy accounting for analysis and validation.