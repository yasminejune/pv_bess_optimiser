# Services: PV Status Management

This module provides services for processing PV telemetry and computing validated system state.

## Purpose

The `pv_status` module transforms raw PV telemetry into validated, constraint-aware state suitable for decision-making. It handles:
- Missing or invalid sensor data
- Hardware constraint enforcement (rated power, export limits)
- Energy estimation from solar radiance when telemetry is unavailable
- Curtailment logic
- Quality flag tracking

## Module: `pv_status.py`

### Function: `estimate_energy_from_radiance`

Pure helper function for estimating PV energy production from solar radiance.

**Signature:**
```python
def estimate_energy_from_radiance(
    solar_radiance_kw_per_m2: float,
    panel_area_m2: float,
    panel_efficiency: float,
    timestep_minutes: int = 15
) -> float
```

**Formula:**
```
Energy (kWh) = solar_radiance × panel_area × efficiency × (timestep_minutes / 60)
```

**Example:**
```python
from src.ors.services.pv_status import estimate_energy_from_radiance

# Estimate energy for 15-minute interval
energy_kwh = estimate_energy_from_radiance(
    solar_radiance_kw_per_m2=0.8,  # 800 W/m²
    panel_area_m2=500.0,            # 500 m² panels
    panel_efficiency=0.18,          # 18% efficient
    timestep_minutes=15             # default
)
# Result: 0.8 × 500 × 0.18 × 0.25 = 18.0 kWh
```

**Validation:**
- `solar_radiance_kw_per_m2` ≥ 0
- `panel_area_m2` > 0
- `0 ≤ panel_efficiency ≤ 1`
- `timestep_minutes` > 0

### Function: `update_pv_state`

Main entrypoint for PV state computation. Processes telemetry and returns validated state.

**Signature:**
```python
def update_pv_state(
    spec: PVSpec,
    telemetry: PVTelemetry,
    timestep_minutes: int = 15,
    *,
    prev_state: PVState | None = None
) -> PVState
```

**Parameters:**
- `spec`: PV system specification (constraints and capabilities)
- `telemetry`: Raw telemetry data
- `timestep_minutes`: Time interval in minutes (default: 15)
- `prev_state`: Optional previous state (reserved for future use, currently unused)

**Returns:**
- `PVState`: Validated and processed state with quality flags

**Processing Logic:**

1. **Generation determination:**
   - If `telemetry.generation_kw` is provided: use it as primary source
   - If `telemetry.generation_kw` is `None`:
     - Add `missing_generation` flag
     - If radiance and panel specs available: estimate energy and derive generation
     - Otherwise: generation = 0

2. **Validation and clamping:**
   - Negative generation → clamp to 0, add `negative_generation_clamped` flag
   - Above rated power → clamp to `spec.rated_power_kw`, add `above_rated_clamped` flag
   - Below minimum → raise to `spec.min_generation_kw`, add `below_min_generation_clamped` flag

3. **Exportable computation:**
   - If `spec.max_export_kw` is `None`: exportable = generation
   - Otherwise: exportable = min(generation, max_export_kw)

4. **Curtailment:**
   - If generation > max_export and curtailment supported: curtailed = generation - max_export
   - If generation > max_export and curtailment NOT supported: add `export_cap_applied_no_curtailment` flag

**Example (telemetry-based):**
```python
from datetime import datetime
from src.ors.domain.models.pv import PVSpec, PVTelemetry
from src.ors.services.pv_status import update_pv_state

# Define system
spec = PVSpec(
    rated_power_kw=100.0,
    max_export_kw=80.0,
    curtailment_supported=True
)

# Process telemetry (default 15-minute timestep)
telemetry = PVTelemetry(
    timestamp=datetime(2026, 1, 15, 12, 0, 0),
    generation_kw=75.0
)

state = update_pv_state(spec, telemetry)

# Results:
# state.generation_kw = 75.0
# state.energy_kwh = 75.0 * (15/60) = 18.75 kWh
# state.exportable_kw = 75.0
# state.curtailed_kw = 0.0
# state.quality_flags = set()
```

**Example (radiance-based estimation):**
```python
# Spec with radiance estimation capability
spec = PVSpec(
    rated_power_kw=100.0,
    max_export_kw=80.0,
    panel_area_m2=500.0,
    panel_efficiency=0.18
)

# Telemetry with missing generation but radiance available
telemetry = PVTelemetry(
    timestamp=datetime(2026, 1, 15, 12, 0, 0),
    generation_kw=None,
    solar_radiance_kw_per_m2=0.8
)

state = update_pv_state(spec, telemetry)

# Results:
# energy_kwh = 0.8 × 500 × 0.18 × 0.25 = 18.0 kWh
# generation_kw = 18.0 / 0.25 = 72.0 kW
# state.estimated_from_radiance = True
# state.quality_flags = {'missing_generation', 'estimated_from_radiance'}
```

**Example (curtailment):**
```python
spec = PVSpec(
    rated_power_kw=100.0,
    max_export_kw=60.0,
    curtailment_supported=True
)

telemetry = PVTelemetry(
    timestamp=datetime(2026, 1, 15, 12, 0, 0),
    generation_kw=85.0  # Exceeds max_export_kw
)

state = update_pv_state(spec, telemetry)

# Results:
# state.generation_kw = 85.0
# state.exportable_kw = 60.0
# state.curtailed_kw = 25.0 (85 - 60)
# state.curtailed = True
```

## Integration Notes

**What this module DOES:**
- Validate and clean PV telemetry
- Enforce hardware constraints
- Estimate generation from radiance when telemetry is missing
- Compute energy produced in a timestep
- Track data quality issues via flags

**What this module DOES NOT do:**
- Interact with battery or grid systems (PV-only scope)
- Fetch weather data or call external APIs
- Perform optimization or scheduling
- Handle data persistence or logging
- Make operational decisions (just provides state)

**Thread Safety:**
All functions are pure or stateless—safe for concurrent use.

## Timestep Convention

The default timestep is **15 minutes** (0.25 hours). This is the standard discretization interval for the ORS. Other timesteps can be specified but must be consistent across the system.

Energy conversion:
- `energy_kwh = power_kw × (timestep_minutes / 60)`
- For 15 minutes: `energy_kwh = power_kw × 0.25`

## For Other Team Members

**How to use in your module:**

1. Import the service:
```python
from src.ors.services.pv_status import update_pv_state
from src.ors.domain.models.pv import PVSpec, PVTelemetry
```

2. Define your PV system once:
```python
pv_spec = PVSpec(rated_power_kw=100.0, max_export_kw=80.0)
```

3. Call `update_pv_state` each decision cycle:
```python
pv_state = update_pv_state(pv_spec, telemetry)
```

4. Use the returned state:
```python
available_pv_power = pv_state.exportable_kw
pv_energy_this_interval = pv_state.energy_kwh
has_quality_issues = len(pv_state.quality_flags) > 0
```

**Do not:**
- Directly modify PVSpec or PVState after creation
- Implement your own PV validation logic—use this service
- Assume generation_kw equals exportable_kw (respect export limits)

---

## Module: `weather_to_pv.py` — PV Power Generation

### Function: `generate_pv_power_for_date_range`

Generates 15-minute PV power output by fetching solar irradiance forecasts
from the Open-Meteo API and applying the full PV model pipeline.

**Signature:**
```python
def generate_pv_power_for_date_range(
    config: PVSiteConfig,
    *,
    client: Any | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
) -> pd.DataFrame
```

**Quick Start:**
```python
from ors.config.pv_config import get_pv_config, SiteType
from ors.services.weather_to_pv import generate_pv_power_for_date_range

config = get_pv_config(SiteType.BURST_1)

# Defaults to now -> now + 48 hours
df = generate_pv_power_for_date_range(config=config)

for _, row in df.iterrows():
    print(f"{row['timestamp_utc']}: gen={row['generation_kw']:.1f} kW")
```

**Return Value — `pd.DataFrame`:**

| Column | Type | Unit | Description |
|---|---|---|---|
| `timestamp_utc` | datetime (tz-aware) | — | UTC timestamp at 15-min intervals |
| `generation_kw` | float | kW | Estimated generation after clamping/constraints |

**Notes:**
- Uses the Open-Meteo **forecast API** with the `ukmo_uk_deterministic_2km` model.
- `start_datetime` defaults to the current UTC time; `end_datetime` defaults to start + 48 hours.
- Requesting data too far in the future raises `WeatherFetcherError`.
- All configuration capacities are in MW; output generation is in kW.
- Panel area is derived automatically: `area = dc_capacity_kw / (1.0 × module_efficiency)`.
- See `pv_generation_README.md` for full details, data flow diagrams, and usage examples.
