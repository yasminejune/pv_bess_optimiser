# Services Layer: Operational Recommendation System

This directory contains the core business logic services for the Operational Recommendation System (ORS). The services layer orchestrates data processing, forecasting, optimization, and integration between different system components.

## Service Architecture Overview

The services layer is organized into specialized modules that handle different aspects of the energy optimization workflow:

```
services/
├── data_loading.py          # Data orchestration for optimization runs  
├── pv_status.py            # PV telemetry processing and state management
├── weather_to_pv.py        # PV power generation forecasting
├── battery/                # Battery management and state processing
├── battery_to_optimization/ # Battery data preparation for optimization
├── optimizer/              # Mathematical optimization engine
├── prediction/             # Price forecasting models and pipelines
├── price_api/              # Historical and live price data retrieval
└── price_inference/        # Real-time price prediction inference
```

## Data Orchestration

### Module: `data_loading.py`

Central data loading orchestrator for optimization runs, supporting multiple data sources and automatic feature preparation.

**Main Functions:**

#### `load_price_data(config, start_datetime, end_datetime, time_step_minutes=15)`
Loads price data based on optimization configuration from multiple sources.

**Supported Sources:**
- `"forecast"`: Live price predictions from ML models
- `"historical"`: Historical price data from CSV files
- `"manual"`: User-defined price profiles

**Returns:**
- `price_dict`: 1-indexed timestep dictionary of prices
- `terminal_price`: Price for valuing terminal battery energy

#### `load_solar_data(config, start_datetime, end_datetime, time_step_minutes=15)`
Loads solar generation data based on PV configuration.

**Supported Sources:**
- `"forecast"`: Weather-based PV generation forecasting
- `"historical"`: Historical solar generation from CSV files
- `"manual"`: User-defined generation profiles

**Returns:** 1-indexed dictionary of solar generation (MW) for each timestep

**Helper Functions:**
- `_load_forecasted_prices()`: Integrate with live price prediction models
- `_load_historical_prices()`: Load and interpolate historical price data
- `_load_manual_prices()`: Process user-defined price profiles
- `_load_forecasted_solar()`: Generate PV forecasts from weather APIs
- `_load_historical_solar()`: Load historical solar generation data
- `_load_manual_solar()`: Process user-defined generation profiles
- `_calculate_terminal_price()`: Compute terminal value for battery energy
- `_interpolate_hourly_to_timesteps()`: Resample data to target resolution

**Helper Functions:**
- `_load_forecasted_prices()`: Integrate with live price prediction models
- `_load_historical_prices()`: Load and interpolate historical price data
- `_load_manual_prices()`: Process user-defined price profiles
- `_load_forecasted_solar()`: Generate PV forecasts from weather APIs
- `_load_historical_solar()`: Load historical solar generation data
- `_load_manual_solar()`: Process user-defined generation profiles
- `_calculate_terminal_price()`: Compute terminal value for battery energy
- `_interpolate_hourly_to_timesteps()`: Resample data to target resolution

## Photovoltaic (PV) Services

### Module: `pv_status.py` — PV State Management

Transforms raw PV telemetry into validated, constraint-aware state suitable for decision-making.

**Key Functions:**

#### `update_pv_state(spec, telemetry, timestep_minutes=15, *, prev_state=None)`
Main entrypoint for PV state computation. Processes telemetry and returns validated state.

**Processing Logic:**
1. **Generation determination:** Uses telemetry or estimates from solar radiance
2. **Validation and clamping:** Enforces rated power, export limits, minimum generation
3. **Exportable computation:** Calculates power available for grid export
4. **Curtailment:** Handles excess generation based on system capabilities
5. **Quality tracking:** Flags missing data, estimates, and constraint violations

**Quality Flags:**
- `missing_generation`: Telemetry generation was None
- `estimated_from_radiance`: Generation derived from solar radiance
- `negative_generation_clamped`: Negative generation clamped to zero
- `above_rated_clamped`: Generation exceeded rated power
- `below_min_generation_clamped`: Generation below minimum threshold
- `export_cap_applied_no_curtailment`: Export capped but curtailment not supported

#### `estimate_energy_from_radiance(solar_radiance_kw_per_m2, panel_area_m2, panel_efficiency, timestep_minutes=15)`
Pure helper function for estimating PV energy production from solar radiance.

**Formula:** `Energy (kWh) = solar_radiance × panel_area × efficiency × (timestep_minutes / 60)`

### Module: `weather_to_pv.py` — PV Power Generation

Generates 15-minute PV power output by fetching solar irradiance forecasts from Open-Meteo API.

#### `generate_pv_power_for_date_range(config, *, client=None, start_datetime=None, end_datetime=None)`
Fetches weather forecasts and applies full PV model pipeline to generate power predictions.

**Returns:** DataFrame with `timestamp_utc` and `generation_kw` columns

## Service Directories

### `battery/` — Battery Energy Storage Management

Comprehensive battery management services for physical simulation and state processing.

**Key Modules:**
- `battery_management.py`: Core battery physics calculations and simulation
- `battery_status.py`: Battery telemetry processing and state validation
- `demo.py`: Example usage and simulation patterns
- `test.py`: Comprehensive unit tests

**Core Capabilities:**
- Battery physics simulation using industry-standard energy equations
- Telemetry validation and state computation from sensor data
- Loss analysis (charging/discharging inefficiency, auxiliary power, self-discharge)
- Configuration management and parameter validation
- CSV export for simulation results

See [battery/README.md](battery/README.md) for detailed documentation.

### `battery_to_optimization/` — Battery-Optimization Interface

Prepares battery data for mathematical optimization, providing the interface between battery simulation and optimization algorithms.

**Key Modules:**
- `battery_inference.py`: Transforms battery states for optimization input
- `test.py`: Integration testing and validation

**Integration Functions:**
- Battery state aggregation and formatting
- Constraint preparation for optimization models
- CSV output formatting for optimization consumption

See [battery_to_optimization/README.md](battery_to_optimization/README.md) for detailed documentation.

### `optimizer/` — Mathematical Optimization Engine

Implements the core mathematical optimization algorithms for energy storage and generation scheduling.

**Key Modules:**
- `optimizer.py`: Primary optimization algorithm implementation
- `integration.py`: Integration helpers and data preparation functions
- `CHANGES.md`: Version history and algorithm updates

**Capabilities:**
- Multi-period optimization for battery charging/discharging decisions
- Price-aware optimization considering time-varying electricity costs
- Constraint handling for battery limits, power ratings, and operational bounds
- Mathematical model construction and solving

**API Functions:**
- `build_model()`: Construct optimization model from input data
- Integration helpers for combining PV forecasts and price predictions

See [optimizer/README.md](optimizer/README.md) for mathematical details and usage patterns.

### `prediction/` — Price Forecasting Models

Machine learning pipeline for electricity price forecasting using multiple model architectures.

**Key Modules:**
- `prediction_model.py`: Core ML model implementations
- `data_pipeline.py`: Feature engineering and data preprocessing
- `train_script.py`: Model training orchestration
- `hyperparameter_search.py`: Automated hyperparameter optimization
- `report_generator.py`: Model performance assessment

**Model Architectures:**
- Temporal Fusion Transformer (TFT) for day-ahead forecasting
- LightGBM for recursive and direct forecasting approaches
- XGBoost for baseline comparisons

See [prediction/README.md](prediction/README.md) for complete model documentation.

### `price_api/` — Price Data Retrieval

Services for fetching electricity price data from external APIs and historical sources.

**Key Modules:**
- `price_api.py`: Live price data fetching and normalization
- `price_api_hist.py`: Historical price data retrieval and processing

**Data Sources:**
- BMRS (Balancing Mechanism Reporting Service) API integration
- Historical price dataset construction
- Real-time price monitoring

**Capabilities:**
- Multi-source price data aggregation
- Data validation and quality checking
- Time series alignment and interpolation

See [price_api/README.md](price_api/README.md) for API documentation and data formats.

### `price_inference/` — Real-time Price Prediction

Live inference engine for real-time electricity price predictions using trained ML models.

**Key Modules:**
- `live_inference.py`: Real-time prediction orchestration  
- `live_data_pipeline.py`: Live data preprocessing and feature engineering
- `__main__.py`: Command-line inference interface

**Capabilities:**
- Real-time price forecasting for optimization decisions
- Live data integration from multiple sources
- Model serving and prediction caching
- Error handling and fallback mechanisms

See [price_inference/README.md](price_inference/README.md) for inference pipeline documentation.

## Service Integration Patterns

### Data Flow Architecture

1. **Data Ingestion**: Services collect data from sensors, APIs, and historical sources
2. **Processing**: Raw data is validated, cleaned, and transformed into domain models
3. **Forecasting**: ML models generate predictions for prices and solar generation
4. **Optimization**: Mathematical algorithms determine optimal control decisions
5. **Integration**: Results are prepared for external systems and control interfaces

### Service Dependencies

```
data_loading.py
├── price_inference/     # For forecast price data
├── weather_to_pv.py     # For forecast solar data
└── price_api/          # For historical price data

optimizer/
├── battery_to_optimization/  # For battery state preparation
├── data_loading.py          # For price and solar data
└── integration.py           # For data combination

battery_to_optimization/
└── battery/                 # For battery state computation

price_inference/
├── prediction/              # For trained ML models
└── price_api/              # For live price data integration
```

### Error Handling and Resilience

Services implement comprehensive error handling:
- **Data Quality**: Validation and quality flags for missing/invalid data
- **Fallback Mechanisms**: Default values and previous state recovery
- **Constraint Enforcement**: Physical and operational limit validation  
- **Logging**: Detailed error tracking and debugging information

### Thread Safety

All services follow immutable data patterns and stateless function design for safe concurrent usage across the system.

## Module Usage Guidelines

**For Optimization Workflows:**
1. Use `data_loading.py` to orchestrate input data collection
2. Configure battery systems via `battery/` services
3. Apply `optimizer/` for mathematical decision-making
4. Integrate results through `battery_to_optimization/`

**For Live Operations:**
1. Use `price_inference/` for real-time price predictions
2. Apply `pv_status` for real-time PV state management  
3. Use `battery/battery_status.py` for real-time battery state
4. Feed results to optimization for operational decisions

**For Historical Analysis:**
1. Use `price_api/` for historical price data retrieval
2. Apply `prediction/` for model training and backtesting
3. Use `optimizer/` for historical optimization simulation
4. Generate reports through service-specific analysis modules
