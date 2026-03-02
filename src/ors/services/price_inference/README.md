# Price Inference Service

This integration module is in charge of performing live predictions using the Weather and Price APIs and feeding their outputs into the Price Prediction Model.

The code allows for several models to be tested via Pickle files.

---

## General Process

1. Fetch weather and electricity price data (live or historical).
2. Run the ETL pipeline used during training.
3. Produces a 96-row (24-hour at 15-minute resolution) price forecast CSV using a pickle file to run models.

Two modes are supported:
- **Live mode** (default): fetches data up to now, forecast starts from the current hour.
- **Historical mode**: pass a `reference_time` to anchor all fetches and the forecast window to a past date, enabling backtesting and evaluation against known prices.

---

## Importing from Other Modules

The package exposes its public functions at the top level via `__init__.py`.

### Run the full pipeline in one call

`run_inference` is the simplest entry point — it fetches live data, runs ETL,
loads the model, predicts, resamples to 15-min resolution, saves the CSV and
returns the results DataFrame.

```python
from datetime import datetime, timezone
from pathlib import Path
from ors.services.price_inference import run_inference

# Use all defaults (MODEL_PATH, HORIZON_HOURS, OUTPUT_PATH) — live mode
results = run_inference()

# Override model path
results = run_inference(
    model_path=Path("models/price_prediction/my_model.pkl"),
    horizon_hours=48,
)

# Historical mode — replay a specific past date for backtesting
results = run_inference(
    reference_time=datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc),
)

# CSV mode — use training CSVs instead of live APIs (reproduces training predictions exactly)
results = run_inference(
    reference_time=datetime(2025, 1, 15, 12, 0, tzinfo=timezone.utc),
    use_csv=True,
)
```

`results` is a DataFrame with columns `Timestamp` and `Price_pred`
(`horizon_hours * 4` rows at 15-minute resolution).

### Import individual functions

If you only need a specific step, import it directly:

```python
# Recommended — short form via __init__.py
from ors.services.price_inference import build_live_merged_dataset
from ors.services.price_inference import load_model, prepare_features_for_inference, select_forecast_rows
```

The long form (direct module path) also works and is equivalent:

```python
# Long form — identical result
from ors.services.price_inference.live_data_pipeline import build_live_merged_dataset
from ors.services.price_inference.live_inference import load_model
```

Use the short form for any new code that depends on this service.

---

## CLI Usage

A. Run with the pre-set model (default, live mode).

```bash
# Run from the project root
python -m ors.services.price_inference.live_inference
```

B. Test a different model.

Pass `--model` with the path to any `.pkl` file:

```bash
# Run from the project root
python -m ors.services.price_inference.live_inference --model models/price_prediction/your_model.pkl
```

C. Historical replay — predict as if it were a specific past date.

Pass `--date` with an ISO datetime (UTC assumed):

```bash
# Run from the project root
python -m ors.services.price_inference.live_inference --date 2025-01-15T12:00
```

D. CSV mode — reproduce training predictions exactly.

Pass `--use-csv` to build features from `Data/` CSVs instead of live APIs.
Useful for backtesting and validating that inference matches training:

```bash
python -m ors.services.price_inference.live_inference --date 2025-01-15T12:00 --use-csv
```

Options can be combined:

```bash
python -m ors.services.price_inference.live_inference \
    --model models/price_prediction/your_model.pkl \
    --date 2025-01-15T12:00 \
    --use-csv
```

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
| `PRICE_LOOKBACK_HOURS` | `48` | Hours of price history fetched; must be >= largest lag step |

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

  Orchestrates the full ETL pipeline (live or historical):
  1. Fetch weather and price data, anchored to `reference_time` if provided
  2. Determine time window from weather and sun only — price is excluded from `end_date`
    to preserve future forecast rows
  3. Apply per-source transforms (normalise, cyclic-encode, solar intensity, hourly aggregation)
  4. Merge all four sources on `Timestamp`
  5. Add lagged features with `drop_na=False` — future rows keep NaN price lags

  Returns a `DataFrame` sorted by `Timestamp`, ready for inference.

---

### `live_inference.py`

| Constant | Value | Purpose |
|---|---|---|
| `MODEL_PATH` | `models/price_prediction/model.pkl` | Trained model (pickle format) |
| `HORIZON_HOURS` | `24` | Hours ahead to predict |
| `LAG_STEPS` | `(1, 2, 3, 6, 12, 24)` | Match training |
| `OUTPUT_PATH` | `data/live_price_predictions.csv` | Output consumed by optimizer |

- `load_model(model_path)`

  Loads the `.pkl` model from disk. Raises `FileNotFoundError` if not found.

- `select_forecast_rows(live_df, horizon_hours, reference_time=None)`

  Filters rows whose `Timestamp` is at or after `reference_time` (or the current hour in
  live mode), then returns the first `horizon_hours` of those rows.

- `prepare_features_for_inference(df, model, target_col)`

  Aligns the live DataFrame to exactly what the model expects:
  1. Drop `Price` and `Timestamp`
  2. Cast `bool` columns to `int`
  3. Fill NaNs with column medians
  4. `reindex(model.feature_names_in_, fill_value=0)` — drops extra columns, adds missing as `0`
  5. Force all columns to numeric dtype

- `run_inference(model_path, horizon_hours, lag_steps, output_path, reference_time=None, use_csv=False, project_root=None)`

  Orchestrates the full pipeline: build dataset → load model → select rows → predict →
  forward-fill to 96 × 15-min slots → save CSV. Pass `reference_time` for historical mode.
  Pass `use_csv=True` to source features from the training CSVs in `Data/` instead of live
  APIs — this reproduces training predictions exactly for any covered timestamp.

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

- **15-minute output is produced by forward-fill**
The current model (xgboost via the pickle file)
  predicts hourly — each value is repeated across its four 15-minute slots at the step 5 of `live_inference.py`.
  
  The pipeline integrates with the trained model
  by running identical ETL functions from `ors.etl.etl` and aligning features to
  `model.feature_names_in_` at inference time.

- **Future price lags are approximated.** Forecast rows have no real price data, so lag columns
  are filled with column medians. This is the main accuracy trade-off of live inference.

- **Boolean columns are cast to `int`.** `IsWeekend`, `IsHoliday`, and `IsWorkingDay` are
  coerced to `1`/`0`. The current model (XGBoost) requires this; keep it for any replacement
  unless the new model handles booleans natively.

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
                    build_live_merged_dataset()                     build_merged_dataset()
                    ← live_data_pipeline.py                         ← prediction/data_pipeline.py
                            │                                                     │
                            └───────────────────┬─────────────────────────────────┘
                                                │  ETL (identical to training)
                                                │  NaN price lags for future rows (expected)
                                                ▼
                                    select_forecast_rows()   ← first 24 rows at or after reference_time
                                                │
                                    prepare_features_for_inference()
                                                │  align to model.feature_names_in_
                                                ▼
                                    model.predict()  →  24 hourly floats
                                                │
                                    forward-fill  →  96 × 15-min slots
                                                ▼
                                    data/live_price_predictions.csv  (columns: Timestamp, Price_pred)  →  Optimizer
```

---