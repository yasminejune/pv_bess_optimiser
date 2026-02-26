# PV Power Generation — Forecast Function

## Overview

`generate_pv_power_for_date_range()` is the primary entrypoint for computing
15-minute PV power generation forecasts. It connects the PV site configuration,
the Open-Meteo 15-minute solar irradiance forecast API, and the power
calculation pipeline into a single call.

## Data Flow

```
PVSiteConfig (MW)
      |
      v
pv_site_config_to_spec()              [pv_converter.py]
  - MW -> kW conversion
  - panel_area_m2 derived from DC capacity & module efficiency
      |
      v
   PVSpec (kW)
      |
      v
generate_pv_power_for_date_range()     [weather_to_pv.py]
  |                    |
  v                    v
solar_radiance_15_mins()   pv_states_from_hourly_weather_df()
  [weather_client.py]        |
  shortwave_radiation        |-> hourly_weather_df_to_pv_telemetry()
  (W/m2 at 15-min intervals) |     W/m2 -> kW/m2 -> PVTelemetry
                              |
                              |-> update_pv_state() for each 15-min step
                              |     estimate generation from radiance
                              |     clamp to rated power
                              |     apply curtailment
                              v
                         list[PVState]
                              |
                              v
                   pd.DataFrame (timestamp_utc, generation_kw)
```

## Usage

### With a Predefined Site Configuration

```python
from ors.config.pv_config import get_pv_config, SiteType
from ors.services.weather_to_pv import generate_pv_power_for_date_range

# Use one of the predefined BURST site configurations
config = get_pv_config(SiteType.BURST_1)  # 65 MW DC / 50 MW AC
# or
config = get_pv_config(SiteType.BURST_2)  # 130 MW DC / 100 MW AC

# Defaults to now -> now + 48 hours
df = generate_pv_power_for_date_range(config=config)
```

### With a Custom Configuration

```python
from ors.config.pv_config import PVSiteConfig

custom_config = PVSiteConfig(
    site_id="My_Site",
    pv_block="PV1",
    pv_capacity_dc_mw=10.0,
    pv_capacity_ac_mw=8.0,
    dc_ac_ratio=1.25,
    module_efficiency=0.20,
    inverter_efficiency=0.98,
    performance_ratio=0.80,
    degradation_per_year=0.5,
    curtailment_threshold_mw=7.5,
    clipping_loss_factor=0.02,
    availability=0.99,
    forced_outage_duration_h=2.0,
)

df = generate_pv_power_for_date_range(config=custom_config)
```

### With Explicit Start/End Times

```python
from datetime import datetime, timezone

start = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
end = datetime(2026, 3, 1, 18, 0, tzinfo=timezone.utc)

df = generate_pv_power_for_date_range(
    config=config,
    start_datetime=start,
    end_datetime=end,
)
```

### Inspecting Results

```python
print(df.head())
#              timestamp_utc  generation_kw
# 0  2026-03-01 10:00:00+00:00        12500.0
# 1  2026-03-01 10:15:00+00:00        14200.0
# 2  2026-03-01 10:30:00+00:00        15800.0
# ...

print(f"Total rows: {len(df)}")
print(f"Peak generation: {df['generation_kw'].max():.1f} kW")
```

## Function Signature

```python
def generate_pv_power_for_date_range(
    config: PVSiteConfig,
    *,
    client: Any | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
) -> pd.DataFrame
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `config` | `PVSiteConfig` | required | PV site configuration (MW units) |
| `client` | `Any \| None` | `None` | Open-Meteo API client; auto-created when `None` |
| `start_datetime` | `datetime \| None` | `None` | Start of window (defaults to current UTC time) |
| `end_datetime` | `datetime \| None` | `None` | End of window (defaults to start + 48 hours) |

## Output

A DataFrame with two columns at 15-minute intervals:

| Column | Type | Unit | Description |
|---|---|---|---|
| `timestamp_utc` | datetime (tz-aware) | — | UTC timestamp of the 15-min interval |
| `generation_kw` | float | kW | Estimated PV generation after clamping and constraints |

## PV Site Configuration

Defined in `src/ors/config/pv_config.py` as a frozen dataclass `PVSiteConfig`.

**Predefined sites** (in the `PV_SITE_CONFIGS` dict):

| Site | DC Capacity | AC Capacity | Curtailment Threshold |
|---|---|---|---|
| `BURST_1` | 65 MW | 50 MW | 48 MW |
| `BURST_2` | 130 MW | 100 MW | 95 MW |

Access via `get_pv_config(SiteType.BURST_1)`.

**Custom sites**: instantiate `PVSiteConfig(...)` directly with your own values
(see example above). All capacity fields are in MW; the pipeline converts to kW
internally.

## Units Summary

| Layer | Power | Energy | Irradiance |
|---|---|---|---|
| PVSiteConfig (input) | MW | — | — |
| Weather API (internal) | — | — | W/m2 |
| DataFrame output | kW | — | — |

## Quality Flags (Internal)

The pipeline uses `PVState` internally which tracks quality flags. These are
not exposed in the output DataFrame but are applied during generation clamping:

| Flag | Meaning |
|---|---|
| `estimated_from_radiance` | Generation was estimated from solar irradiance |
| `missing_generation` | No direct generation measurement was available |
| `above_rated_clamped` | Raw generation exceeded rated AC capacity |
| `below_min_generation_clamped` | Generation was below minimum threshold |
| `negative_generation_clamped` | Negative generation was clamped to zero |

## Limitations

- Uses the Open-Meteo **forecast API** with the `ukmo_uk_deterministic_2km` model.
  Forecast availability is limited to roughly 5-10 days ahead; requesting data
  too far in the future raises `WeatherFetcherError`.
- Location is fixed to the default in `weather_client.DEFAULT_PARAMS` (Cumbria, UK).
- Panel area is derived from DC capacity at Standard Test Conditions (1.0 kW/m2
  irradiance). This is standard industry practice but does not account for
  site-specific panel layouts.
