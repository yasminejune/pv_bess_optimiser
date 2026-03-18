# Domain Models: PV System & Battery Energy Storage

This module contains domain models for photovoltaic (PV) and battery energy storage system (BESS) specifications, telemetry, and state representation.

## Purpose

The domain models provide type-safe, validated data structures for:
- **PV system specifications** (hardware constraints and capabilities)
- **PV telemetry inputs** (sensor measurements or forecasts)
- **PV derived system state** (processed outputs with quality flags)
- **Battery system specifications** (physical characteristics and operational constraints)
- **Battery telemetry inputs** (real-time sensor data from battery management systems)
- **Battery derived system state** (validated state with quality flags and estimations)

## Models

### PVSpec

Defines the physical characteristics and operational constraints of a PV installation.

**Fields:**
- `rated_power_kw` (float): Maximum rated power output in kW
- `max_export_kw` (float | None): Maximum power that can be exported to grid (kW). `None` means unlimited.
- `min_generation_kw` (float): Minimum generation threshold (kW), default 0.0
- `curtailment_supported` (bool): Whether system can curtail excess generation, default `True`
- `panel_area_m2` (float | None): Total panel surface area (m²). Optional, used for radiance-based estimation.
- `panel_efficiency` (float | None): Panel efficiency ratio (0–1). Optional, used for radiance-based estimation.

**Validation:**
- `rated_power_kw` must be positive
- `max_export_kw` must be non-negative (if provided)
- `min_generation_kw` must be non-negative
- `panel_area_m2` must be positive (if provided)
- `panel_efficiency` must be between 0 and 1 (if provided)

**Example:**
```python
from src.ors.domain.models.pv import PVSpec

# Basic spec without radiance estimation
spec = PVSpec(
    rated_power_kw=100.0,
    max_export_kw=80.0,
    min_generation_kw=0.0,
    curtailment_supported=True
)

# Spec with radiance estimation capabilities
spec_with_radiance = PVSpec(
    rated_power_kw=100.0,
    max_export_kw=80.0,
    panel_area_m2=500.0,      # 500 m² of panels
    panel_efficiency=0.18     # 18% efficiency
)
```

### PVTelemetry

Raw input data from sensors, forecasts, or estimates.

**Fields:**
- `timestamp` (datetime): Measurement or forecast timestamp
- `generation_kw` (float | None): Instantaneous measured PV output (kW). `None` indicates missing telemetry.
- `solar_radiance_kw_per_m2` (float | None): Solar radiance (kW/m²). Optional, used for estimation when `generation_kw` is missing.

**Note on units:**
- Solar radiance is measured in **kW/m²** (kilowatts per square meter)
- The default system timestep is **15 minutes**

**Example:**
```python
from datetime import datetime
from src.ors.domain.models.pv import PVTelemetry

# Telemetry with direct generation measurement
telemetry = PVTelemetry(
    timestamp=datetime(2026, 1, 15, 12, 0, 0),
    generation_kw=45.2
)

# Telemetry with radiance (for estimation when generation is unavailable)
telemetry_radiance = PVTelemetry(
    timestamp=datetime(2026, 1, 15, 12, 15, 0),
    generation_kw=None,
    solar_radiance_kw_per_m2=0.75  # 750 W/m²
)
```

### PVState

Validated and processed PV system state with derived values and quality indicators.

**Fields:**
- `timestamp` (datetime): State timestamp
- `generation_kw` (float): Final cleaned/derived generation power (kW)
- `energy_kwh` (float): Energy produced this timestep (kWh)
- `curtailed_kw` (float): Amount of power curtailed (kW)
- `curtailed` (bool): Whether curtailment is active
- `exportable_kw` (float): Power available for export (kW)
- `exportable_kwh` (float): Energy available for export this timestep (kWh)
- `estimated_from_radiance` (bool): Whether generation was estimated from radiance
- `quality_flags` (set[str]): Quality issue or processing flags

**Quality Flags:**
- `missing_generation`: Telemetry generation was None
- `estimated_from_radiance`: Generation derived from solar radiance
- `negative_generation_clamped`: Negative generation value was clamped to zero
- `above_rated_clamped`: Generation exceeded rated power and was clamped
- `below_min_generation_clamped`: Generation below minimum was raised to minimum
- `export_cap_applied_no_curtailment`: Export capped but curtailment not supported

**Example:**
```python
from src.ors.domain.models.pv import PVState

# State is typically created by pv_status service, but can be constructed:
state = PVState(
    timestamp=datetime(2026, 1, 15, 12, 0, 0),
    generation_kw=45.2,
    energy_kwh=11.3,          # 45.2 kW * 0.25 hours
    curtailed_kw=0.0,
    curtailed=False,
    exportable_kw=45.2,
    exportable_kwh=11.3,
    estimated_from_radiance=False,
    quality_flags=set()
)
```

## Energy Estimation from Radiance

When `generation_kw` is unavailable but `solar_radiance_kw_per_m2` and panel specifications (`panel_area_m2`, `panel_efficiency`) are provided, energy can be estimated using:

```
Energy (kWh) = solar_radiance (kW/m²) × panel_area (m²) × efficiency × time_interval (hours)
```

For the default **15-minute timestep**: `time_interval = 15/60 = 0.25 hours`

This estimation is performed automatically by the `pv_status` service when telemetry is missing.

## Usage in the System

These models are used by:
1. **Configuration**: Define PV system specs at startup
2. **Telemetry ingestion**: Parse and validate incoming sensor data
3. **State computation**: The `pv_status` service transforms telemetry into validated state
4. **Downstream modules**: Battery and market logic consume PVState (but PV models remain independent)

---

# Battery Energy Storage System Models

## Models

### BatterySpec

Defines the physical characteristics and operational constraints of a battery energy storage system.

**Fields:**
- `rated_power_mw` (float): Maximum rated power input/output in MW
- `energy_capacity_mwh` (float): Total energy capacity in MWh
- `min_soc_percent` (float): Minimum allowable state of charge percentage (0-100), default 10.0
- `max_soc_percent` (float): Maximum allowable state of charge percentage (0-100), default 90.0
- `charge_efficiency` (float): Charging efficiency (0-1), default 0.97
- `discharge_efficiency` (float): Discharging efficiency (0-1), default 0.97
- `auxiliary_power_mw` (float): Auxiliary power consumption in MW, default 0.5
- `self_discharge_rate_per_hour` (float): Self-discharge rate per hour (0-1), default 0.0005

**Validation:**
- `rated_power_mw` and `energy_capacity_mwh` must be positive
- `min_soc_percent` and `max_soc_percent` must be between 0 and 100
- `min_soc_percent` must be ≤ `max_soc_percent`
- `charge_efficiency` and `discharge_efficiency` must be between 0 and 1
- `auxiliary_power_mw` and `self_discharge_rate_per_hour` must be non-negative

**Computed Properties:**
- `min_energy_mwh`: Minimum energy in MWh (capacity × min_soc_percent / 100)
- `max_energy_mwh`: Maximum energy in MWh (capacity × max_soc_percent / 100)

**Example:**
```python
from src.ors.domain.models.battery import BatterySpec

# Standard utility-scale battery system
spec = BatterySpec(
    rated_power_mw=100.0,
    energy_capacity_mwh=400.0,  # 4-hour duration
    min_soc_percent=10.0,
    max_soc_percent=90.0,
    charge_efficiency=0.97,
    discharge_efficiency=0.97,
    auxiliary_power_mw=0.5,
    self_discharge_rate_per_hour=0.0005
)

print(f"Usable energy: {spec.max_energy_mwh - spec.min_energy_mwh} MWh")
```

### BatteryTelemetry

Raw input data from real-time sensors or battery management systems.

**Fields:**
- `timestamp` (datetime): Measurement or report timestamp
- `current_energy_mwh` (float | None): Current stored energy in MWh. `None` indicates missing telemetry.
- `current_soc_percent` (float | None): Current state of charge percentage (0-100). `None` indicates missing telemetry.
- `current_power_mw` (float | None): Current power flow in MW. Positive = charging, Negative = discharging, 0 = idle.
- `operating_mode` (Literal["charging", "discharging", "idle"] | None): Current operating mode. `None` indicates unknown.
- `is_available` (bool): Whether battery is available for optimization control, default `True`

**Validation:**
- `current_soc_percent` must be between 0 and 100 (if provided)

**Helper Methods:**
- `get_energy_from_soc(spec)`: Convert SOC percentage to energy using battery specification
- `get_soc_from_energy(spec)`: Convert energy to SOC percentage using battery specification

**Example:**
```python
from datetime import datetime
from src.ors.domain.models.battery import BatteryTelemetry

# Telemetry with direct energy measurement
telemetry = BatteryTelemetry(
    timestamp=datetime(2026, 3, 11, 14, 30, 0),
    current_energy_mwh=250.0,
    current_soc_percent=62.5,  # Should be consistent with energy
    current_power_mw=50.0,     # Charging at 50 MW
    operating_mode="charging",
    is_available=True
)

# Telemetry with missing data (will require estimation)
telemetry_partial = BatteryTelemetry(
    timestamp=datetime(2026, 3, 11, 14, 45, 0),
    current_energy_mwh=None,    # Missing - will estimate from SOC
    current_soc_percent=65.0,
    current_power_mw=None,      # Missing - will estimate
    operating_mode=None,        # Missing - will estimate from power
    is_available=True
)
```

### BatteryState

Validated and processed battery state with derived values and quality indicators.

**Fields:**
- `timestamp` (datetime): State timestamp
- `energy_mwh` (float): Final validated stored energy (MWh)
- `soc_percent` (float): Final validated state of charge percentage (0-100)
- `power_mw` (float): Current power flow (MW). Positive = charging, Negative = discharging, 0 = idle
- `operating_mode` (Literal["charging", "discharging", "idle"]): Current operating mode
- `is_available` (bool): Whether battery is available for optimization
- `estimated_values` (set[str]): Set of field names that were estimated rather than measured
- `quality_flags` (set[str]): Quality flags indicating any issues or processing notes

**Quality Flags:**
- `missing_energy_and_soc`: Both energy and SOC telemetry unavailable
- `energy_estimated_from_soc`: Energy derived from SOC percentage
- `energy_soc_mismatch`: Provided energy and SOC values are inconsistent (>1% difference)
- `using_previous_state`: Used previous state as fallback for missing data
- `defaulted_to_50_percent_soc`: No data available, defaulted to 50% SOC for safety
- `energy_clamped_to_minimum`: Energy was below minimum and clamped to `min_energy_mwh`
- `energy_clamped_to_maximum`: Energy was above maximum and clamped to `max_energy_mwh`
- `missing_power_data`: Power measurement unavailable
- `power_clamped_to_rated`: Power exceeded rated limits and was clamped
- `mode_power_mismatch`: Reported operating mode inconsistent with power measurement
- `mode_estimated_from_power`: Operating mode estimated from power flow
- `unrealistic_energy_change`: Energy change between states exceeds physical limits
- `invalid_mode_defaulted_to_idle`: Invalid operating mode was corrected to idle

**Estimated Values:**
- `energy_mwh`: Energy was estimated rather than directly measured
- `soc_percent`: SOC was estimated rather than directly measured
- `power_mw`: Power was estimated (typically defaults to 0.0)
- `operating_mode`: Operating mode was estimated from power flow

**Properties:**
- `has_quality_issues`: Whether this state has any quality issues
- `has_estimated_values`: Whether this state contains estimated values

**Example:**
```python
from src.ors.domain.models.battery import BatteryState

# High-quality state with no issues
state = BatteryState(
    timestamp=datetime(2026, 3, 11, 14, 30, 0),
    energy_mwh=250.0,
    soc_percent=62.5,
    power_mw=50.0,
    operating_mode="charging",
    is_available=True,
    estimated_values=set(),
    quality_flags=set()
)

# State with quality issues and estimations
state_with_issues = BatteryState(
    timestamp=datetime(2026, 3, 11, 14, 45, 0),
    energy_mwh=260.0,  # Estimated from SOC
    soc_percent=65.0,
    power_mw=0.0,      # Estimated (missing)
    operating_mode="idle",  # Estimated from power
    is_available=True,
    estimated_values={"energy_mwh", "power_mw", "operating_mode"},
    quality_flags={"energy_estimated_from_soc", "missing_power_data", "mode_estimated_from_power"}
)

print(f"Has issues: {state_with_issues.has_quality_issues}")
print(f"Has estimates: {state_with_issues.has_estimated_values}")
```

## Battery Usage Patterns and Integration

**State Computation Flow:**
1. **Telemetry Input**: Raw sensor data (`BatteryTelemetry`)
2. **Validation & Processing**: The `battery_status` service transforms telemetry
3. **Quality Assessment**: Missing data estimation and cross-validation
4. **State Output**: Clean, validated `BatteryState` for downstream use

**Energy vs. SOC Handling:**
- Energy measurement is prioritized over SOC for accuracy
- If both are provided, consistency is checked (>1% difference flags `energy_soc_mismatch`)
- If only SOC is available, energy is calculated using battery capacity
- Missing both results in fallback to previous state or safe default (50% SOC)

**Power and Operating Mode:**
- Power measurement determines actual charging/discharging activity
- Operating mode is cross-validated against power flow
- Inconsistencies result in power-based mode determination (power is trusted)
- Missing power data defaults to 0.0 MW (idle state)

**Constraint Enforcement:**
- Energy is clamped to `min_energy_mwh` and `max_energy_mwh` bounds
- Power is clamped to `±rated_power_mw` limits
- All violations are tracked via quality flags

## Battery Model Integration Notes

**What battery models provide:**
- Type-safe data structures for battery specifications and state
- Comprehensive validation of physical constraints
- Quality tracking for data reliability assessment
- Helper methods for energy/SOC conversions

**What battery models do NOT handle:**
- Battery physics simulations (charging/discharging calculations)
- Optimization decisions or control logic
- Data persistence or communication with actual hardware
- Time-series state transitions or forecasting

**Thread Safety:**
All models are immutable once created—safe for concurrent use across services.

**Integration with Services:**
- Configuration: `BatterySpec` defines system capabilities
- Telemetry ingestion: `BatteryTelemetry` validates incoming sensor data
- State processing: `battery_status` service creates `BatteryState` from telemetry
- Optimization: Battery and market services consume `BatteryState` for decisions
