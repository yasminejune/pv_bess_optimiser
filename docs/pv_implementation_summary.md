# PV Configuration Implementation Summary

## Overview
Implemented a comprehensive PV (photovoltaic) system configuration module to accommodate site-specific parameters for BURST deployment scenarios.

## Changes Made

### 1. New Configuration Module
**File:** `src/ors/config/pv_config.py`
- Created `SiteType` enum with 2 BURST site types
- Created `PVSiteConfig` dataclass with full validation
- Implemented `PV_SITE_CONFIGS` dictionary containing experimental values
- Added helper functions: `get_pv_config()` and `list_available_sites()`

**Configuration Values:**
- **Site_Burst_1**: 65 MW DC / 50 MW AC (1.3 DC/AC ratio)
- **Site_Burst_2**: 130 MW DC / 100 MW AC (1.3 DC/AC ratio)

All sites include:
- Module efficiency: 21%
- Inverter efficiency: 98.5%
- Availability: 99.5%
- Site-specific performance ratios, degradation rates, curtailment thresholds, and clipping loss factors

### 2. Enhanced Domain Model
**File:** `src/ors/domain/models/pv.py`
- Extended `PVSpec` dataclass with additional fields:
  - `dc_capacity_kw`
  - `ac_capacity_kw`
  - `dc_ac_ratio`
  - `inverter_efficiency`
  - `performance_ratio`
  - `degradation_per_year`
  - `clipping_loss_factor`
  - `availability`
  - `forced_outage_duration_h`
- All new fields are optional to maintain backward compatibility
- Added comprehensive validation for all new parameters

### 3. Configuration Converter Utility
**File:** `src/ors/utils/pv_converter.py`
- Created `pv_site_config_to_spec()` function
- Converts `PVSiteConfig` to `PVSpec` domain model
- Handles MW to kW conversions automatically
- Maps all configuration parameters to domain model fields

### 4. Comprehensive Tests
**File:** `tests/test_pv_config.py`
- Test cases covering:
  - Configuration retrieval for both BURST sites
  - String-based site lookup
  - Invalid site type handling
  - Site listing functionality
  - Config to PVSpec conversion
  - Validation of configuration parameters
- All tests pass ✓

### 5. Documentation
**File:** `docs/pv_configuration.md`
- Complete usage guide with examples
- Site comparison tables
- Integration examples with existing services
- Extension instructions

### 6. Example Scripts

**File:** `scripts/example_pv_config_usage.py`
- Demonstrates basic configuration usage
- Shows conversion to domain models
- Prints detailed configuration summaries

**File:** `scripts/export_pv_configs.py`
- Exports configurations to CSV format
- Exports configurations to JSON format
- Generates markdown tables for documentation

## Usage Examples

### Get Configuration
```python
from src.ors.config.pv_config import SiteType, get_pv_config

config = get_pv_config(SiteType.BURST_1)
print(f"DC/AC Ratio: {config.dc_ac_ratio}")
```

### Convert to Domain Model
```python
from src.ors.utils.pv_converter import pv_site_config_to_spec

spec = pv_site_config_to_spec(config)
print(f"Rated Power: {spec.rated_power_kw} kW")
```

### Use with Existing Services
```python
from src.ors.services.pv_status import update_pv_state
from src.ors.domain.models.pv import PVTelemetry

telemetry = PVTelemetry(timestamp=now, generation_kw=50000.0)
state = update_pv_state(spec, telemetry, timestep_minutes=15)
```

## Key Design Decisions

1. **Configuration as Code**: Used Python dataclasses for type safety and validation
2. **Separate Config Layer**: Kept configuration separate from domain models for flexibility
3. **Optional Fields**: New PVSpec fields are optional to maintain backward compatibility
4. **Unit Conversions**: Converter handles MW↔kW conversions automatically
5. **Validation**: Comprehensive parameter validation at configuration creation time
6. **Extensibility**: Easy to add new site types by extending the enum and dictionary

## Testing Results
- ✓ All configuration tests pass
- ✓ All existing PV service tests pass (backward compatible)

## Files Created
1. `src/ors/config/__init__.py`
2. `src/ors/config/pv_config.py`
3. `src/ors/utils/__init__.py`
4. `src/ors/utils/pv_converter.py`
5. `tests/test_pv_config.py`
6. `scripts/example_pv_config_usage.py`
7. `scripts/export_pv_configs.py`
8. `docs/pv_configuration.md`

## Files Modified
1. `src/ors/domain/models/pv.py` - Enhanced PVSpec with additional optional fields

## Backward Compatibility
✓ All existing code continues to work without modifications
✓ New fields in PVSpec are optional
✓ Existing tests pass without changes
✓ No breaking changes to public APIs

## Next Steps
The configuration system is ready to use. Suggested next steps:
1. Integrate PV configurations into optimization models
2. Use site-specific parameters in generation forecasting
3. Apply degradation models based on site characteristics
4. Implement curtailment logic using site thresholds
5. Consider time-dependent degradation calculations
