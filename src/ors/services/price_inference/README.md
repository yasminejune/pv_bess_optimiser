# Price Inference Service

This integration module is in charge of performing live predictions using the Weather and Price APIs and feeding their outputs into the Price Prediction Model.

Two model backends are supported:

- **LightGBM ForecasterRecursive** (default) — saved as `.joblib` by the training pipeline. Loaded from `LGBM_MODEL_DIR`. Runs at native 15-min resolution; no resampling required.
- **XGBoost** (legacy) — saved as `model.pkl`. Loaded from `MODEL_PATH`. Generates hourly predictions that are forward-filled to 15-min resolution.

---

## General Process

1. Fetch weather and electricity price data (live or historical).
2. Run the ETL pipeline used during training.
3. Produce a 96-row (24-hour at 15-minute resolution) price forecast CSV.

Two modes are supported:
- **Live mode** (default): fetches data up to now, forecast starts from the current 15-min slot.
- **Historical mode**: pass a `reference_time` to anchor all fetches and the forecast window to a past date, enabling backtesting and evaluation against known prices.

---

## Prerequisites — Training the LGBM model

Before running LGBM inference, the model directory must contain at least one `.joblib` file.
Train the model from the repo root:

```bash
python -m ors.services.prediction.train_script --model-name lgbm_recursive
```

This writes files matching the pattern:

```
models/price_prediction/lgbm_recursive_single_model/
    recursive_single_model_<YYYYMMDD_HHMMSS>.joblib
    recursive_single_model_<YYYYMMDD_HHMMSS>_meta.joblib
```

---

## CLI Usage

### LGBM model (default)

A. Auto-detect — uses the most recent `.joblib` file in `LGBM_MODEL_DIR` (falls back to XGBoost if the directory is missing or empty).

```bash
# Run from the project root
python -m ors.services.price_inference.live_inference
```

B. Explicitly point to the LGBM model directory.

```bash
python -m ors.services.price_inference.live_inference \
    --model models/price_prediction/lgbm_recursive_single_model
```

C. Explicitly point to a specific `.joblib` file.

```bash
python -m ors.services.price_inference.live_inference \
    --model models/price_prediction/lgbm_recursive_single_model/recursive_single_model_20260306_123625.joblib
```

D. LGBM historical replay — predict as if it were a specific past date.

```bash
python -m ors.services.price_inference.live_inference --date 2025-01-15T12:00
```

E. LGBM with CSV mode — reproduce training predictions exactly (bypasses live APIs).

```bash
python -m ors.services.price_inference.live_inference \
    --model models/price_prediction/lgbm_recursive_single_model \
    --date 2025-01-15T12:00 \
    --use-csv
```

### XGBoost model (legacy)

F. Run with the pre-set XGBoost model.

```bash
python -m ors.services.price_inference.live_inference --model models/price_prediction/model.pkl
```

G. Test a different XGBoost model.

```bash
python -m ors.services.price_inference.live_inference --model models/price_prediction/your_model.pkl
```

Options can be combined:

```bash
python -m ors.services.price_inference.live_inference \
    --model models/price_prediction/your_model.pkl \
    --date 2025-01-15T12:00 \
    --use-csv
```

---

## Importing from Other Modules

The package exposes its public functions at the top level via `__init__.py`.

### Run the full pipeline in one call

`run_inference` is the simplest entry point — it auto-detects the model type, fetches live data, runs ETL, predicts, and saves the CSV.

```python
from datetime import datetime, timezone
from pathlib import Path
from ors.services.price_inference import run_inference

# Use all defaults — auto-detects LGBM, runs live
results = run_inference()

# Explicitly use the LGBM model directory
results = run_inference(
    model_path=Path("models/price_prediction/lgbm_recursive_single_model"),
)

# Explicitly use a specific LGBM .joblib file
results = run_inference(
    model_path=Path("models/price_prediction/lgbm_recursive_single_model/recursive_single_model_20260306_123625.joblib"),
)

# Historical mode — replay a specific past date for backtesting
results = run_inference(
    reference_time=datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc),
)

# CSV mode — reproduce training predictions exactly
results = run_inference(
    reference_time=datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc),
    use_csv=True,
)
```

`results` is a DataFrame with columns `Timestamp` and `Price_pred`
(96 rows at 15-minute resolution for a 24-hour horizon).

### Import individual functions

```python
# Recommended — short form via __init__.py
from ors.services.price_inference import build_live_lgbm_dataset
from ors.services.price_inference import find_latest_lgbm_model
from ors.services.price_inference import load_model, prepare_features_for_inference, select_forecast_rows
```

The long form (direct module path) also works and is equivalent:

```python
# Long form — identical result
from ors.services.price_inference.live_data_pipeline import build_live_lgbm_dataset
from ors.services.price_inference.live_inference import find_latest_lgbm_model, load_model
```

Use the short form for any new code that depends on this service.

---

## Modules

| Module | Responsibility |
|---|---|
| `live_data_pipeline.py` | Fetch live API data and run ETL to produce a model-ready DataFrame |
| `live_inference.py` | Load the model, run predictions, resample to 15-min, save output CSV |

---

## Functions per Module

### `live_data_pipeline.py`

| Constant | Value | Purpose |
|---|---|---|
| `PRICE_LOOKBACK_HOURS` | `48` | Hours of price history fetched for XGBoost; must be >= largest lag step |
| `LGBM_PRICE_LOOKBACK_HOURS` | `170` | Hours of price history fetched for LGBM; must cover model `window_size` (168 h) plus buffer |

- `_fetch_live_weather(past_hours, forecast_days, reference_time=None)`

  In **live mode** (`reference_time=None`): calls Open-Meteo Forecast API with the same
  parameters as offline training, returning `(hourly_df, daily_df)` covering `past_hours`
  of history and `forecast_days` ahead.

  In **historical mode**: `reference_time` is accepted but the weather client always
  calls the forecast API regardless. Historical weather replay is reserved for a future
  implementation.

- `_fetch_live_price(lookback_hours, ..., reference_time=None)`

  Calls four BMRS endpoints (MID price, ITSDO, INDO, INDDEM demand), outer-merges on
  `ts_utc`, and returns a DataFrame with columns: `timestamp`, `price`, `demand_itsdo`,
  `demand_indo`, `demand_inddem`.
  When `reference_time` is provided it is used as the end of the fetch window instead of
  `now`, enabling historical replay. Returns an empty DataFrame with the correct schema if
  all endpoints are unavailable.

- `build_live_merged_dataset(past_hours, forecast_days, lag_steps, ..., reference_time=None)`

  Orchestrates the full ETL pipeline for the **XGBoost / legacy path** (live or historical):
  1. Fetch weather and price data, anchored to `reference_time` if provided
  2. Determine time window from weather and sun only — price is excluded from `end_date`
    to preserve future forecast rows
  3. Apply per-source transforms (normalise, cyclic-encode, solar intensity, hourly aggregation)
  4. Merge all four sources on `Timestamp`
  5. Add lagged features with `drop_na=False` — future rows keep NaN price lags

  Returns a `DataFrame` sorted by `Timestamp`, ready for inference.

- `build_live_lgbm_dataset(forecast_steps, past_hours, ..., reference_time=None)`

  Orchestrates the full ETL pipeline for the **LGBM ForecasterRecursive path** (live or historical):
  1. Fetch weather (`past_hours` history + enough forecast days to cover `forecast_steps`)
  2. Fetch price data (past only — future rows will have `price = NaN`)
  3. Append a sentinel row at `now + forecast_steps × 15 min` so the preprocessing pipeline
     builds a `full_index` that covers the entire future exog horizon
  4. Run `preprocess_raw_data(..., drop_missing_price=False)` — the same function used during
     training — so feature columns are identical to what the fitted `ForecasterRecursive` expects

  Returns a `DataFrame` sorted by `Timestamp`.  Past rows have a valid `price` column; future rows
  have `price = NaN`.  All exogenous feature columns are populated for both windows.

  `past_hours` defaults to `LGBM_PRICE_LOOKBACK_HOURS` (170 h) to cover the model's full
  `window_size` of 672 steps × 15 min = 168 h.

---

### `live_inference.py`

| Constant | Value | Purpose |
|---|---|---|
| `LGBM_MODEL_DIR` | `models/price_prediction/lgbm_recursive_single_model` | Directory searched for LGBM `.joblib` files |
| `LGBM_HORIZON_STEPS` | `96` | 15-min steps predicted in one shot (= 24 h); must match training `HORIZON` |
| `MODEL_PATH` | `models/price_prediction/model.pkl` | Legacy XGBoost model path |
| `HORIZON_HOURS` | `24` | Hours ahead to predict (XGBoost path) |
| `LAG_STEPS` | `(1, 2, 3, 6, 12, 24)` | Lag offsets in hours — must match XGBoost training |
| `OUTPUT_PATH` | `data/live_price_predictions.csv` | Output consumed by optimizer |

#### LGBM functions

- `find_latest_lgbm_model(model_dir) → (model_path, meta_path | None)`

  Scans `model_dir` for files matching `recursive_single_model_<timestamp>.joblib` (excludes
  `_meta` files), sorts by filename, and returns the most recently written one together with
  its companion `_meta.joblib` path (or `None` if the meta file is absent).
  Raises `FileNotFoundError` if the directory does not exist or contains no matching files,
  with a hint to run the training script.

- `_load_lgbm_meta(model_path) → dict | None`

  Loads the companion `<stem>_meta.joblib` file that lives alongside the model file.
  Returns the meta dictionary (which includes `feature_cols`, training timestamps, etc.)
  or `None` if the file does not exist.  The meta file is the preferred source for
  `feature_cols`; if absent, feature columns are derived from the live DataFrame.

- `_is_lgbm_forecaster(model) → bool`

  Returns `True` if the loaded model is a skforecast `ForecasterRecursive`; used internally
  to branch between the LGBM and XGBoost inference paths.

- `_extract_lgbm_inputs(df, feature_cols, window_size, forecast_steps, reference_time=None)`

  Extracts the three inputs that `ForecasterRecursive.predict()` requires from a merged
  DataFrame:
  1. `last_window` — `pd.Series` of the last `window_size` observed prices before `reference_time`
     (or now), forward-filled to handle 30-min API gaps, with `RangeIndex` `0..window_size-1`.
  2. `exog` — `pd.DataFrame` of exogenous feature columns for the next `forecast_steps` rows,
     reindexed to exactly `feature_cols` (missing columns filled with `0.0`).
     Index is offset by `window_size` so it starts at `window_size` — required by skforecast's
     contiguity check (`exog.index[0]` must equal `last_window.index[-1] + 1`).
  3. `forecast_index` — `pd.DatetimeIndex` of 15-min timestamps for the forecast window,
     starting at `reference_time` (or now) floored to the nearest 15 min.

  Works for both live datasets (future rows have `price = NaN`) and CSV/backtest datasets
  (all rows have real prices — future rows are still selected by timestamp).

#### Shared / XGBoost functions

- `load_model(model_path)`

  Loads a model from disk.  When `model_path` is a **directory**, delegates to
  `find_latest_lgbm_model` and loads the most recent `.joblib` file automatically.
  Supports `.joblib` (LGBM / skforecast) and `.pkl` (XGBoost) formats.
  Raises `FileNotFoundError` if the file or directory does not exist.

- `select_forecast_rows(live_df, horizon_hours, reference_time=None)`

  Filters rows whose `Timestamp` is at or after `reference_time` (or the current hour in
  live mode), then returns the first `horizon_hours` of those rows.
  Used by the XGBoost path only; the LGBM path uses `_extract_lgbm_inputs` instead.

- `prepare_features_for_inference(df, model, target_col)`

  Aligns the live DataFrame to exactly what the XGBoost model expects:
  1. Drop `Price` and `Timestamp`
  2. Cast `bool` columns to `int`
  3. Fill NaNs with column medians
  4. `reindex(model.feature_names_in_, fill_value=0)` — drops extra columns, adds missing as `0`
  5. Force all columns to numeric dtype

- `run_inference(model_path, horizon_hours, lag_steps, output_path, reference_time=None, use_csv=False, project_root=None)`

  Orchestrates the full pipeline and auto-detects the model type:

  **LGBM path** (when model resolves to a `.joblib` file):
  1. Build dataset via `build_live_lgbm_dataset` (or `build_merged_dataset` when `use_csv=True`)
  2. Load `feature_cols` from the meta file (preferred) or derive from the DataFrame
  3. Call `_extract_lgbm_inputs` to obtain `last_window`, `exog`, and `forecast_index`
  4. Call `model.predict(steps, last_window, exog)` — produces native 15-min predictions

  **XGBoost path** (when model resolves to a `.pkl` file):
  1. Build dataset via `build_live_merged_dataset`
  2. Select forecast rows with `select_forecast_rows`
  3. Call `prepare_features_for_inference` then `model.predict`
  4. Forward-fill hourly predictions to 96 × 15-min slots

  Both paths save results to `output_path` and return a DataFrame with columns `Timestamp`
  and `Price_pred`.

- `main()`

  CLI entry point. Accepts `--model`, `--date`, and `--use-csv` arguments and delegates to
  `run_inference`.

---

## Notes

- **API predictions will not match `Prediction/Models/.../predictions.csv`**
  This is expected and not a bug. The training predictions were generated from static CSV
  snapshots in `Data/`. The APIs serve *living datasets* that are retrospectively revised:

  - **Weather** — Open-Meteo's historical data is based on ERA5 reanalysis, a meteorological
    model that is periodically rerun with corrections. Querying the same date weeks later can
    return slightly different temperature, humidity, or solar values.
  - **Price/demand** — BMRS electricity settlement data goes through multiple revision runs
    (initial → interim → final settlement), so the price for a past timestamp may have changed
    since the training CSV was downloaded.
  - **Lag features compound the difference** — because `price_lag_1h`, `price_lag_24h` etc.
    are derived from the base values, a small raw difference fans out across many feature
    columns, which is why the prediction gap can be significant (e.g. ~4 £/MWh for the same
    timestamp).

  Use `--use-csv` to bypass the APIs and build features from the original training CSVs,
  which reproduces training predictions exactly. This is useful for validation but not for
  production, where you want the most up-to-date data.

- **LGBM produces native 15-min predictions.**
  The `ForecasterRecursive` is trained at 15-min resolution and predicts 96 steps (24 h) in
  a single call.  No forward-fill resampling is applied — each slot has its own distinct
  prediction.  The XGBoost legacy path still forward-fills from hourly to 15-min.

- **`exog` index must be contiguous with `last_window`.**
  `skforecast` enforces `exog.index[0] == last_window.index[-1] + 1`.  Both arrays
  use integer `RangeIndex`; `last_window` occupies `0..window_size-1`, so `exog` must
  start at `window_size`.  `_extract_lgbm_inputs` applies
  `exog_df.index = exog_df.index + window_size` after `reset_index(drop=True)` to
  satisfy this constraint.  Without this offset skforecast raises:
  `ValueError: To make predictions exog must start one step ahead of last_window.`

- **`window_size` must be covered by `LGBM_PRICE_LOOKBACK_HOURS`.**
  The default LGBM config uses `max_lag=672` steps × 15 min = 168 h.  `LGBM_PRICE_LOOKBACK_HOURS`
  is set to 170 h to provide a small buffer against minor API gaps.  If you retrain with a larger
  `max_lag`, increase this constant accordingly.

- **Future price lags are approximated (XGBoost path only).**
  XGBoost forecast rows have no real price data, so price lag columns are filled with column
  medians.  The LGBM `ForecasterRecursive` handles this internally via its recursive prediction
  mechanism — it uses its own previous predictions to fill lags when stepping forward in time.

- **Boolean columns are cast to `int`.**
  `IsWeekend`, `IsHoliday`, and `IsWorkingDay` are coerced to `1`/`0` in
  `prepare_features_for_inference` (XGBoost path).  The LGBM path does not need this because
  `_extract_lgbm_inputs` passes only the exog feature columns that the `ForecasterRecursive`
  was trained with.

- **Warnings during execution are harmless.** A pandas `PerformanceWarning` from ETL and any
  serialisation warning from the model loader do not affect results.

---

## Data Flow

```
Open-Meteo [Forecast API]                                          Data/price_data.csv
      +  BMRS Elexon API (live or historical window)               Data/historical_hourly_2025.csv
            │                                                       Data/historical_daily_2025.csv
            │  reference_time=None  →  window ends at now                        │
            │  reference_time=<dt>  →  window ends at <dt>                       │ use_csv=True
                                    │ use_csv=False                               │
              ┌─────────────────────┴──┐                         build_merged_dataset()
              │ LGBM path              │ XGBoost path            ← prediction/data_pipeline.py
              ▼                        ▼                                          │
   build_live_lgbm_dataset()  build_live_merged_dataset()                        │
   ← live_data_pipeline.py    ← live_data_pipeline.py                            │
            │                          │                                          │
            │  past_hours=170 h        │  past_hours=48 h                        │
            │  price=NaN for future    │  price=NaN for future                   │
            └──────────────┬───────────┘                                         │
                           └───────────────────────────────────────────────────┘
                                                │  ETL (identical to training)
                                                ▼
                         ┌──────────────────────────────────────────────┐
                         │ LGBM ForecasterRecursive path                │
                         │                                              │
                         │  _extract_lgbm_inputs()                      │
                         │    → last_window  (window_size price steps)  │
                         │    → exog         (96 feature rows)          │
                         │    → forecast_index                          │
                         │                                              │
                         │  model.predict(steps=96,                     │
                         │               last_window, exog)             │
                         │    → 96 × 15-min floats (native resolution)  │
                         └──────────────────────────────────────────────┘
                         ┌──────────────────────────────────────────────┐
                         │ XGBoost / legacy path                        │
                         │                                              │
                         │  select_forecast_rows()                      │
                         │    → first 24 rows at or after reference     │
                         │                                              │
                         │  prepare_features_for_inference()            │
                         │    → align to model.feature_names_in_        │
                         │                                              │
                         │  model.predict()  →  24 hourly floats        │
                         │                                              │
                         │  forward-fill  →  96 × 15-min slots          │
                         └──────────────────────────────────────────────┘
                                                │
                                                ▼
                         data/live_price_predictions.csv
                         (columns: Timestamp, Price_pred)  →  Optimizer
```

---
