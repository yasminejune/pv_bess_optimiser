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
