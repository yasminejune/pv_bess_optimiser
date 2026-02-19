# PV Configuration Quick Reference

## Import Statements

```python
# Get site configurations
from src.ors.config.pv_config import SiteType, get_pv_config, list_available_sites

# Convert to domain models
from src.ors.utils.pv_converter import pv_site_config_to_spec

# Use domain models
from src.ors.domain.models.pv import PVSpec, PVTelemetry, PVState
from src.ors.services.pv_status import update_pv_state
```

## Quick Examples

### 1. Get a Site Configuration
```python
# By enum
config = get_pv_config(SiteType.BURST_1)

# By string
config = get_pv_config("Site_Burst_1")
```

### 2. Access Configuration Values
```python
config = get_pv_config(SiteType.BURST_1)
print(config.pv_capacity_dc_mw)        # 65.0 MW
print(config.pv_capacity_ac_mw)        # 50.0 MW
print(config.dc_ac_ratio)              # 1.3
print(config.module_efficiency)        # 0.21
print(config.curtailment_threshold_mw) # 48.0 MW
```

### 3. Convert to Domain Model
```python
config = get_pv_config(SiteType.BURST_1)
spec = pv_site_config_to_spec(config)
print(spec.rated_power_kw)    # 50000.0 kW (50 MW)
print(spec.max_export_kw)     # 48000.0 kW (48 MW)
```

### 4. List All Sites
```python
sites = list_available_sites()
# ['Site_Burst_1', 'Site_Burst_2']
```

### 5. Use with PV Status Service
```python
from datetime import datetime

config = get_pv_config(SiteType.BURST_1)
spec = pv_site_config_to_spec(config)

telemetry = PVTelemetry(
    timestamp=datetime.now(),
    generation_kw=45000.0,  # 45 MW
    solar_radiance_kw_per_m2=0.8
)

state = update_pv_state(spec, telemetry, timestep_minutes=15)
print(f"Generation: {state.generation_kw} kW")
print(f"Curtailed: {state.curtailed_kw} kW")
print(f"Exportable: {state.exportable_kw} kW")
```

## Site Comparison Table

| Site | Profile | DC (MW) | AC (MW) | Ratio | Perf Ratio | Degrad | Curtail (MW) |
|------|---------|---------|---------|-------|------------|--------|--------------|
| Burst_1 | Burst | 65 | 50 | 1.3 | 0.82 | 0.5% | 48 |
| Burst_2 | Burst | 130 | 100 | 1.3 | 0.82 | 0.5% | 95 |

## Parameter Descriptions

| Parameter | Unit | Description |
|-----------|------|-------------|
| `pv_capacity_dc_mw` | MW | DC capacity of PV array |
| `pv_capacity_ac_mw` | MW | AC capacity of inverter |
| `dc_ac_ratio` | - | DC/AC capacity ratio |
| `module_efficiency` | 0-1 | PV module efficiency (21%) |
| `inverter_efficiency` | 0-1 | Inverter efficiency (98.5%) |
| `performance_ratio` | 0-1 | Overall system performance |
| `degradation_per_year` | % | Annual degradation rate |
| `curtailment_threshold_mw` | MW | Power curtailment threshold |
| `clipping_loss_factor` | 0-1 | Clipping loss factor |
| `availability` | 0-1 | System availability (99.5%) |
| `forced_outage_duration_h` | hours | Expected outage duration (1h) |

## Common Use Cases

### Iterate Over All Sites
```python
for site_type in SiteType:
    config = get_pv_config(site_type)
    print(f"{config.site_id}: {config.pv_capacity_ac_mw} MW")
```

### Export Configuration
```python
# Run export script
# PYTHONPATH=. python scripts/export_pv_configs.py
```

## Testing

```bash
# Run PV configuration tests
pytest tests/test_pv_config.py -v

# Run all PV-related tests
pytest tests/ -k pv -v

# Run example usage
PYTHONPATH=. python scripts/example_pv_config_usage.py
```

## Files Reference

- **Config**: `src/ors/config/pv_config.py`
- **Converter**: `src/ors/utils/pv_converter.py`
- **Domain Model**: `src/ors/domain/models/pv.py`
- **Tests**: `tests/test_pv_config.py`
- **Docs**: `docs/pv_configuration.md`
- **Examples**: `scripts/example_pv_config_usage.py`
- **Export**: `scripts/export_pv_configs.py`
