# PV Configuration System

This document describes the PV (photovoltaic) system configuration infrastructure for the ORS BESS Optimizer.

## Overview

The PV configuration system provides site-specific parameters for BURST deployment scenarios. It supports:

- **2 BURST site types**: Different capacity configurations
- **Comprehensive PV parameters**: DC/AC capacity, efficiencies, degradation, curtailment thresholds
- **Type-safe configuration**: Using dataclasses with validation
- **Easy conversion**: Between configuration and domain models

## Site Types

### Burst Profiles (Higher DC/AC Ratio)
- **Site_Burst_1**: 65 MW DC / 50 MW AC (1.3 ratio)
- **Site_Burst_2**: 130 MW DC / 100 MW AC (1.3 ratio)

## Key Parameters

All configurations include:

| Parameter | Description | Unit |
|-----------|-------------|------|
| `pv_capacity_dc_mw` | DC capacity of PV array | MW |
| `pv_capacity_ac_mw` | AC capacity of inverter | MW |
| `dc_ac_ratio` | DC to AC capacity ratio | - |
| `module_efficiency` | PV module efficiency | 0-1 |
| `inverter_efficiency` | Inverter efficiency | 0-1 |
| `performance_ratio` | Overall system performance | 0-1 |
| `degradation_per_year` | Annual degradation rate | % |
| `curtailment_threshold_mw` | Power curtailment threshold | MW |
| `clipping_loss_factor` | Clipping loss factor | 0-1 |
| `availability` | System availability factor | 0-1 |
| `forced_outage_duration_h` | Expected outage duration | hours |

## Usage Examples

### Basic Usage

```python
from src.ors.config.pv_config import SiteType, get_pv_config

# Get configuration for a specific site
config = get_pv_config(SiteType.BURST_1)
print(f"DC Capacity: {config.pv_capacity_dc_mw} MW")
print(f"AC Capacity: {config.pv_capacity_ac_mw} MW")
print(f"DC/AC Ratio: {config.dc_ac_ratio}")
```

### Using String Identifiers

```python
config = get_pv_config("Site_Burst_1")
```

### List Available Sites

```python
from src.ors.config.pv_config import list_available_sites

sites = list_available_sites()
# ['Site_Burst_1', 'Site_Burst_2']
```

### Convert to Domain Model

```python
from src.ors.config.pv_config import get_pv_config, SiteType
from src.ors.utils.pv_converter import pv_site_config_to_spec

# Get configuration
config = get_pv_config(SiteType.BURST_1)

# Convert to PVSpec domain model
spec = pv_site_config_to_spec(config)

# Use in domain logic
print(f"Rated Power: {spec.rated_power_kw} kW")  # 50000 kW
print(f"Max Export: {spec.max_export_kw} kW")    # 48000 kW
```

### Integration with PV Status Service

```python
from datetime import datetime
from src.ors.config.pv_config import get_pv_config, SiteType
from src.ors.utils.pv_converter import pv_site_config_to_spec
from src.ors.domain.models.pv import PVTelemetry
from src.ors.services.pv_status import update_pv_state

# Get site configuration
config = get_pv_config(SiteType.BURST_1)
spec = pv_site_config_to_spec(config)

# Create telemetry data
telemetry = PVTelemetry(
    timestamp=datetime.now(),
    generation_kw=75000.0,  # 75 MW
    solar_radiance_kw_per_m2=0.8
)

# Update PV state
state = update_pv_state(spec, telemetry, timestep_minutes=15)

print(f"Generation: {state.generation_kw} kW")
print(f"Curtailed: {state.curtailed_kw} kW")
print(f"Exportable: {state.exportable_kw} kW")
```

## Burst Profile Characteristics

| Metric | Burst Profiles |
|--------|---------------|
| DC/AC Ratio | 1.3 |
| Performance Ratio | 0.82 |
| Degradation/Year | 0.5% |
| Clipping Loss | 3% |

**Key Insights:**
- **Burst profiles** have higher DC/AC ratios for aggressive oversizing, suitable for capturing peak generation
- Both configurations maintain 99.5% availability and use 21% efficient modules with 98.5% efficient inverters

## File Structure

```
src/ors/
├── config/
│   ├── __init__.py
│   └── pv_config.py          # Site configurations and enums
├── domain/
│   └── models/
│       └── pv.py              # Enhanced with new parameters
├── services/
│   └── pv_status.py           # PV state update service
└── utils/
    ├── __init__.py
    └── pv_converter.py        # Config to domain model conversion

tests/
└── test_pv_config.py          # Configuration tests

scripts/
└── example_pv_config_usage.py # Usage examples
```

## Validation

All configuration parameters are validated on creation:

```python
from src.ors.config.pv_config import PVSiteConfig

# This will raise ValueError
invalid_config = PVSiteConfig(
    site_id="Test",
    pv_block="PV1",
    pv_capacity_dc_mw=-10.0,  # ❌ Must be positive
    # ... other params
)
```

## Testing

Run the test suite:

```bash
# Run all PV-related tests
pytest tests/ -k pv -v

# Run only configuration tests
pytest tests/test_pv_config.py -v

# Run example script
PYTHONPATH=. python scripts/example_pv_config_usage.py
```

## Extension

To add a new site configuration:

1. Add a new enum value to `SiteType` in `pv_config.py`
2. Add the configuration to `PV_SITE_CONFIGS` dictionary
3. Add test cases in `tests/test_pv_config.py`

```python
class SiteType(str, Enum):
    # ... existing sites
    BURST_3 = "Site_Burst_3"

PV_SITE_CONFIGS = {
    # ... existing configs
    SiteType.BURST_3: PVSiteConfig(
        site_id="Site_Burst_3",
        pv_block="PV1",
        pv_capacity_dc_mw=100.0,
        pv_capacity_ac_mw=80.0,
        # ... other params
    ),
}
```

## See Also

- [PV Status Service Documentation](../domain/models/pv.py)
- [Domain Models](../domain/models/)
- [Example Usage Script](../../scripts/example_pv_config_usage.py)
