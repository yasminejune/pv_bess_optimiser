# Price Inference Service

This integration module is in charge of performing live predictions using the Weather and Price APIs and feeding their outputs into the Price Prediction Model.

The code allows for several models to be tested via Pickle files.

---

## General Process

1. Fetch live weather and electricity price data.
2. Run the ETL pipeline used during training.
3. Produces a 96-row (24-hour at 15-minute resolution) price forecast CSV using a pickle file to run models.

---

## CLI Usage

A. Run with the pre-set model (default).

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

---

## Importing from Other Modules

The package exposes its public functions at the top level via `__init__.py`.

### Run the full pipeline in one call

`run_inference` is the simplest entry point — it fetches live data, runs ETL,
loads the model, predicts, resamples to 15-min resolution, saves the CSV and
returns the results DataFrame.

```python
from ors.services.price_inference import run_inference

# Use all defaults (MODEL_PATH, HORIZON_HOURS, OUTPUT_PATH)
results = run_inference()

# Override any parameter
results = run_inference(
    model_path=Path("models/price_prediction/my_model.pkl"),
    horizon_hours=48,
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

- `_fetch_live_weather(past_hours, forecast_days)`

  Calls Open-Meteo Forecast API with the same parameters and variables as offline training.
  Returns `(hourly_df, daily_df)` covering `past_hours` of history and `forecast_days` ahead.

-  `_fetch_live_price(lookback_hours, ...)`

  Calls four BMRS endpoints (MID price, ITSDO, INDO, INDDEM demand), outer-merges on
  `ts_utc`, and returns a DataFrame with columns: `timestamp`, `price`, `demand_itsdo`,
  `demand_indo`, `demand_inddem`.
  Returns an empty DataFrame with the correct schema if all endpoints are unavailable.

- `build_live_merged_dataset(past_hours, forecast_days, lag_steps, ...)`

  Orchestrates the full live ETL pipeline:
  1. Fetch weather and price data
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
| `OUTPUT_PATH` | `Data/live_price_predictions.csv` | Output consumed by optimizer |

- `load_model(model_path)`

  Loads the `.pkl` model from disk. Raises `FileNotFoundError` if not found.

- `select_forecast_rows(live_df, horizon_hours)`

  Returns the last `horizon_hours` rows of the sorted DataFrame. No timestamp comparison
  needed — the weather forecast always places future rows at the end.

- `prepare_features_for_inference(df, model, target_col)`

  Aligns the live DataFrame to exactly what the model expects:
  1. Drop `Price` and `Timestamp`
  2. Cast `bool` columns to `int`
  3. Fill NaNs with column medians
  4. `reindex(model.feature_names_in_, fill_value=0)` — drops extra columns, adds missing as `0`
  5. Force all columns to numeric dtype

- `main()`

  Runs the full pipeline
  
  build dataset → load model → select rows → predict →
  forward-fill to 96 × 15-min slots → save CSV.

---

## Notes

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
Open-Meteo Forecast API  +  BMRS Elexon API
            │
    build_live_merged_dataset()   ← live_data_pipeline.py
            │  ETL (identical to training)
            │  NaN price lags for future rows (expected)
            ▼
    select_forecast_rows()        ← last 24 rows
            │
    prepare_features_for_inference()
            │  align to model.feature_names_in_
            ▼
    model.predict()  →  24 hourly floats
            │
    forward-fill  →  96 × 15-min slots
            ▼
    Data/live_price_predictions.csv  (columns: Timestamp, Price_pred)  →  Optimizer
```

---