# Domain Models: PV System

This module contains domain models for photovoltaic (PV) system specifications, telemetry, and state representation.

## Purpose

The PV domain models provide type-safe, validated data structures for:
- **PV system specifications** (hardware constraints and capabilities)
- **Telemetry inputs** (sensor measurements or forecasts)
- **Derived system state** (processed outputs with quality flags)

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
