# BMRS Price & Demand Pipeline (`price_api.py`)

A modular pipeline to download BMRS (Elexon) data, normalize it, build **15-minute** features, and produce a final dataset for price/demand modeling.

Main file: `src/ors/services/price_api.py`

---

## What this script does

The pipeline:

1. Downloads data from multiple BMRS endpoints (MID, ITSDO, INDO, INDDEM, and optional FREQ).
2. Tries multiple request strategies (`from/to`, `publish window`, `settlement window`, etc.) per endpoint.
3. Automatically selects the best path + parameter style combination per chunk.
4. Normalizes timestamps to UTC.
5. Builds numeric features and resamples to 15 minutes.
6. Joins all endpoint features into a common master time grid.
7. Creates calendar features and `target_price` lags.
8. Saves outputs (CSV and optional Parquet).

---

## Module architecture

### 1) Config and models

- `PipelineConfig`: full runtime configuration.
- `EndpointSpec`: endpoint definition (paths, chunk size, mode, parameter styles).

### 2) HTTP / request layer

- `make_session()`: `requests` session with retries.
- `request_with_params(...)`: generic BMRS request.
- Endpoint-specific request functions:
  - `request_price_mid_chunk(...)`
  - `request_demand_itsdo_chunk(...)`
  - `request_demand_indo_chunk(...)`
  - `request_demand_inddem_chunk(...)`
  - `request_freq_chunk(...)`

### 3) Parameter builders

- `params_from_to(...)`
- `params_publish_window(...)`
- `params_settlement_window(...)`
- `params_settlement_single(...)`
- `params_price_mid(...)` (optional MID providers filter)

### 4) Data transformation

- `attach_ts(...)`: finds/builds a timestamp column.
- `numeric_ts_frame(...)`: numeric conversion + aggregation by `ts`.
- `mid_feature_frame(...)`: MID-specific feature logic (price/volume by provider).
- `resample_15m(...)`: frequency normalization.
- `choose_target_price_col(...)`: target column selection.
- `add_calendar_and_lags(...)`: calendar + lag features.

### 5) Orchestration

- `fetch_endpoint_robust(...)`: robust endpoint/chunk fetch.
- `build_bmrs_dataset_15m_all(...)`: full end-to-end pipeline.

---

## Supported endpoints

Default endpoints:

- `price_mid`
  - Paths: `/balancing/pricing/market-index`, `/datasets/MID`
  - Mode: `half_hour`
  - Chunk: 7 days
  - Params: `from_to` (forced for MID)

- `demand_itsdo`
  - Path: `/datasets/ITSDO`
  - Mode: `half_hour`
  - Chunk: 1 day
  - Params: `publish_window`, `from_to`, `settlement_window`

- `demand_indo`
  - Paths: `/demand/outturn/stream`, `/datasets/INDO`
  - Mode: `half_hour`
  - Chunk: 1 day
  - Params: `settlement_window`, `settlement_single`, `publish_window`, `from_to`

- `demand_inddem`
  - Path: `/datasets/INDDEM`
  - Mode: `half_hour`
  - Chunk: 1 day
  - Params: `from_to`, `publish_window`, `settlement_window`

Optional:

- `freq` if `enable_freq=True`.

---

## Requirements

Minimum:

- Python 3.10+ (recommended 3.11/3.12)
- `requests`
- `pandas`
- `numpy`
- `python-dateutil`

If saving Parquet:

- `pyarrow` (or `fastparquet`)

---

## Installation

### Option A (recommended): editable install

```bash
pip install -e .
```

---

# Historical Price Data Builder (`price_api_hist.py`)

A specialized module for building comprehensive historical price datasets from BMRS APIs with advanced chunking, error handling, and data validation capabilities.

## Overview

The `price_api_hist` module provides enterprise-grade historical data collection by:

1. **Chunked Data Fetching**: Efficiently processes large time ranges by splitting into manageable chunks
2. **Multi-Endpoint Coordination**: Coordinates fetching across multiple BMRS endpoints (MID, demand sources)
3. **Robust Error Handling**: Continues processing when individual chunks fail with comprehensive error logging
4. **Data Quality Validation**: Validates temporal consistency and data completeness across all sources
5. **Standardized Output**: Produces analysis-ready CSV datasets with consistent 15-minute temporal resolution

## Core Classes

### `HistoryConfig` — Configuration Management

Runtime configuration for building historical price datasets with comprehensive parameter control.

**Key Attributes:**
- `base_url`: BMRS base URL for API requests
- `start_utc`: ISO-8601 start datetime (inclusive)
- `end_utc`: ISO-8601 end datetime (exclusive)  
- `output_csv`: Output CSV filename (default: `price_data.csv`)
- `freq`: Master dataset frequency (must be "15min" for standard use-case)
- `mid_chunk_days`: Chunk size for MID (price) requests (default: 7 days)
- `demand_chunk_days`: Chunk size for demand requests (default: 1 day)
- `mid_providers`: MID provider list for selective data fetching
- `timeout`: HTTP request timeout in seconds
- `continue_on_error`: Whether to continue processing when chunks fail

### `PriceHistoryBuilder` — Historical Data Pipeline

Main orchestration class that manages the complete historical data collection workflow.

**Key Methods:**

#### `__init__(cfg: HistoryConfig)`
Initializes builder with configuration and creates master time index.
- Sets up output directory structure
- Creates 15-minute frequency master index for data alignment
- Validates configuration parameters

#### `build_price_data()`
Executes the complete historical data collection pipeline.

**Pipeline Stages:**
1. **MID Price Data**: Fetches market index prices using chunked requests
2. **Demand Data Collection**: Fetches multiple demand series (ITSDO, INDO, INDDEM)
3. **Temporal Alignment**: Aligns all data sources to common 15-minute index
4. **Resampling**: Converts half-hourly data to 15-minute resolution using forward-fill
5. **Quality Validation**: Checks data completeness and temporal consistency
6. **CSV Export**: Outputs standardized dataset for analysis

#### `_safe_call(fn, *args, **kwargs)`
Robust function execution wrapper with comprehensive error handling.
- Catches and logs exceptions without stopping pipeline
- Provides detailed error context for debugging
- Enables fault-tolerant data collection across large time ranges

## Integration with Main Price API

The `price_api_hist` module works in coordination with `price_api.py`:

- **price_api.py**: Live data fetching and real-time pipeline processing
- **price_api_hist.py**: Historical data collection and batch processing for model training

**Shared Components:**
- Common BMRS API wrapper functions (`fetch_mid_price`, `fetch_*_demand`)
- Standardized UTC timestamp handling (`to_dt_utc`)
- Session management and HTTP retry logic (`make_session`)
- Consistent data transformation patterns

## Usage Examples

### Basic Historical Data Collection
```python
from pathlib import Path
from price_api_hist import HistoryConfig, PriceHistoryBuilder

# Configure historical data collection
config = HistoryConfig(
    base_url="https://data.elexon.co.uk/bmrs/api/v1",
    start_utc="2024-01-01T00:00:00Z",
    end_utc="2024-12-31T23:59:59Z",
    out_dir="historical_data",
    output_csv="price_data_2024.csv",
    mid_chunk_days=7,      # Weekly price chunks
    demand_chunk_days=1,   # Daily demand chunks
    continue_on_error=True
)

# Build historical dataset
builder = PriceHistoryBuilder(config)
result = builder.build_price_data()

print(f"Dataset created: {result['output_path']}")
print(f"Records collected: {result['total_records']}")
```

### Large-Scale Historical Collection
```python
# Multi-year collection with robust error handling
config = HistoryConfig(
    base_url="https://data.elexon.co.uk/bmrs/api/v1",
    start_utc="2020-01-01T00:00:00Z",
    end_utc="2025-01-01T00:00:00Z",
    output_csv="price_data_5year.csv",
    mid_chunk_days=14,     # Larger chunks for efficiency
    demand_chunk_days=2,   # Larger demand chunks
    timeout=120,           # Extended timeout for large requests
    continue_on_error=True # Continue despite individual chunk failures
)

builder = PriceHistoryBuilder(config)
result = builder.build_price_data()

# Review any errors that occurred
if result['errors']:
    print(f"Completed with {len(result['errors'])} chunk errors")
    for error in result['errors']:
        print(f"  - {error['chunk']}: {error['message']}")
```

## Data Quality Features

### Temporal Consistency
- **15-minute Resolution**: All outputs standardized to 15-minute intervals
- **UTC Normalization**: Consistent timezone handling across all data sources
- **Forward Fill**: Half-hourly data resampled with `ffill(limit=1)` for intermediate points
- **Gap Detection**: Automatic identification of missing time periods

### Error Resilience  
- **Chunk-Level Isolation**: Individual chunk failures don't stop overall collection
- **Retry Logic**: Built-in HTTP retry mechanisms for transient failures
- **Progress Tracking**: Detailed logging of collection progress and error recovery
- **Data Validation**: Verification of collected data against expected schemas

### Output Standardization
- **Consistent Columns**: Standardized column naming across all BMRS endpoints
- **Missing Value Handling**: Systematic approach to gaps and missing data
- **Metadata Preservation**: Maintains data provenance and collection timestamps
- **Format Compatibility**: Outputs compatible with downstream ML pipeline requirements

## Performance Considerations

### Chunking Strategy
- **MID Data**: 7-day chunks balance API limits with request efficiency
- **Demand Data**: 1-2 day chunks accommodate higher frequency demand endpoints
- **Memory Management**: Streaming processing prevents memory overflow for large datasets
- **Rate Limiting**: Respects BMRS API rate limits with configurable delays

### Scalability Features
- **Parallel Processing**: Can be extended for concurrent endpoint fetching
- **Resume Capability**: Designed for checkpoint/resume functionality in future versions  
- **Storage Optimization**: CSV output optimized for downstream processing efficiency
- **Memory Efficiency**: Processes data in chunks rather than loading entire datasets

## Integration Notes

The historical data builder integrates seamlessly with the broader ORS pipeline:
- **Model Training**: Provides historical data for price prediction model training
- **Backtesting**: Enables historical optimization performance evaluation
- **Feature Engineering**: Outputs compatible with ETL pipeline feature engineering
- **Validation**: Historical data serves as ground truth for model validation
