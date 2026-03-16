# Price Inference Service — Change Log

Branch: `regressor_pred`

The goal of this work was to wire the LightGBM `ForecasterRecursive` model
(trained in `ors.services.prediction`) into the live inference pipeline, which
previously only supported the legacy XGBoost backend.

---

## Files changed

### `__init__.py` — **new file**

Created a package-level `__init__.py` that re-exports the public API so callers
can use the short import form:

```python
from ors.services.price_inference import run_inference, build_live_lgbm_dataset
```

Exported symbols: `build_live_lgbm_dataset`, `build_live_merged_dataset`,
`find_latest_lgbm_model`, `load_model`, `prepare_features_for_inference`,
`run_inference`, `select_forecast_rows`.

---

### `live_data_pipeline.py` — **LGBM dataset builder added**

**Before:** contained only `build_live_merged_dataset` (XGBoost path), which
built an hourly feature DataFrame with lagged price columns.

**After:** retains the XGBoost path unchanged and adds a new LGBM-specific
builder plus shared helpers.

**Added constants:**

| Constant | Value | Purpose |
|---|---|---|
| `LGBM_PRICE_LOOKBACK_HOURS` | `170` | Hours of price history fetched for LGBM; covers the model's `window_size` (168 h) plus buffer |
| `_MID_MAX_WINDOW_HOURS` | `168` | Max window per BMRS MID API call (7-day limit) |

**Added `_chunked_fetch(fetch_fn, start_utc, end_utc, max_hours)`**

BMRS MID price endpoint rejects requests spanning more than 7 days.  This
helper slices a wider time window into compliant chunks, calls `fetch_fn` on
each, and returns a deduplicated, concatenated result sorted by `ts_utc`.
Used by `_fetch_live_price` when `lookback_hours > 168`.

**Added `build_live_lgbm_dataset(forecast_steps, past_hours, ..., reference_time)`**

Builds the merged DataFrame for LGBM inference:

1. Calls `_fetch_live_weather` with enough `forecast_days` to cover
   `forecast_steps × 15 min` (minimum 3 days, calculated as
   `(forecast_steps // 96) + 2`).
2. Calls `_fetch_live_price` for `past_hours` of history (default 170 h).
3. Appends a sentinel row at `now + forecast_steps × 15 min` so that
   `preprocess_raw_data` builds a `full_index` extending into the forecast
   window — without it, future exog rows would be missing from the output.
4. Runs `preprocess_raw_data(..., drop_missing_price=False)` — the same
   function used during training — ensuring feature columns match exactly
   what the fitted `ForecasterRecursive` expects.

Returns a DataFrame sorted by `Timestamp`.  Past rows have a valid `price`
column; future rows have `price = NaN` and are used as exogenous features.

---

### `live_inference.py` — **LGBM inference path added + bug fix**

**Before:** contained only the XGBoost path.  `run_inference` called
`build_live_merged_dataset`, `select_forecast_rows`,
`prepare_features_for_inference`, and `model.predict` in that order.

**After:** auto-detects the model type and branches.

**Added functions:**

- `find_latest_lgbm_model(model_dir) → (model_path, meta_path | None)`

  Scans `model_dir` for files matching
  `recursive_single_model_<timestamp>.joblib` (excludes `_meta` files), sorts
  by filename, and returns the most recently written one with its companion
  meta path.  Raises `FileNotFoundError` with a training hint if none are
  found.

- `_load_lgbm_meta(model_path) → dict | None`

  Loads `<stem>_meta.joblib` alongside the model file.  Returns the dict
  (containing `feature_cols`, hyperparameters, training timestamps) or `None`
  if absent.  When present, `feature_cols` from meta is used instead of
  deriving it from the live DataFrame.

- `_is_lgbm_forecaster(model) → bool`

  Returns `True` when `type(model).__name__ == "ForecasterRecursive"`.  Used
  in `run_inference` to select the inference branch.

- `_extract_lgbm_inputs(df, feature_cols, window_size, forecast_steps, reference_time)`

  Splits the merged DataFrame into the three inputs that
  `ForecasterRecursive.predict()` requires:
  - `last_window` — last `window_size` observed price values before
    `reference_time` (or now), forward-filled to handle 30-min API gaps,
    with a `RangeIndex` starting at `0`.
  - `exog` — the next `forecast_steps` rows reindexed to `feature_cols`
    (missing columns filled with `0.0`).
  - `forecast_index` — `DatetimeIndex` of 15-min slots for the output.

**Bug fixed in `_extract_lgbm_inputs`:**

`skforecast` validates that `exog.index[0] == last_window.index[-1] + 1`.
Both `last_window` and `exog` were produced with `.reset_index(drop=True)`,
so both started at `0`.  With `window_size=480`, `last_window` occupied
indices `0..479` and `exog` also started at `0`, causing:

```
ValueError: To make predictions `exog` must start one step ahead of
`last_window`.
  `last_window` ends at : 479.
  `exog` starts at : 0.
  Expected index : 480
```

**Fix** (`live_inference.py:349`):

```python
exog_df = future_df.reindex(columns=feature_cols, fill_value=0.0).reset_index(drop=True)
# skforecast requires exog index to start one step after last_window ends.
# last_window has index 0..window_size-1, so exog must start at window_size.
exog_df.index = exog_df.index + window_size
```

**Updated `load_model`:** accepts a directory as `model_path`; delegates to
`find_latest_lgbm_model` to resolve the most recent `.joblib` file.

**Updated `run_inference`:**

- Accepts the optional `--model` argument (directory or file).
- When model resolves to `.joblib`, runs the LGBM path:
  1. `build_live_lgbm_dataset` (or `build_merged_dataset` with `use_csv=True`)
  2. Load `feature_cols` from meta file (fallback: derive from DataFrame)
  3. `_extract_lgbm_inputs` → `last_window`, `exog`, `forecast_index`
  4. `model.predict(steps, last_window, exog)` — native 15-min output
- When model resolves to `.pkl`, runs the unchanged XGBoost path.

**Added CLI flags:**

| Flag | Default | Purpose |
|---|---|---|
| `--model` | auto | Path to `.joblib`/`.pkl` file or LGBM model directory |
| `--date` | now | ISO reference datetime for historical replay |
| `--use-csv` | `False` | Build features from training CSVs instead of live APIs |

---

### `README.md` — **new file**

Added full documentation covering prerequisites, CLI usage examples (A–G),
importable API, per-module function reference, data flow diagram, and notes on
expected API vs training prediction divergence.

---

## How to run

```bash
# 1. Install deps (lightgbm, skforecast must be present)
pip install -e .

# 2. Train the LGBM model first (if not already done)
python -m ors.services.prediction.train_script --model-name lgbm_recursive

# 3. Run live inference
python -m ors.services.price_inference.live_inference \
    --model models/price_prediction/lgbm_recursive_single_model
```

Output is written to `data/live_price_predictions.csv` (96 rows, columns
`Timestamp` and `Price_pred`).

---

## Known issues / next steps

- **Missing exog columns** (`demand_forecast`, `wind_generation`,
  `solar_generation`, `margin_daily_forecast`) are filled with `0.0` at
  inference time.  These columns correspond to BMRS forecast endpoints not yet
  wired into `live_data_pipeline.py`.  Adding them would improve accuracy on
  the first 24-hour horizon where demand and generation forecasts are available.

- **`last_window` forward-fill** — the BMRS MID price API returns data at
  30-min resolution.  After reindexing to 15-min, intermediate slots are
  forward-filled from the preceding 30-min value.  This is a reasonable
  approximation but not the same as having true 15-min settlement prices.

- **Weather historical replay** — `reference_time` is accepted by
  `_fetch_live_weather` but the Open-Meteo client always calls the live
  Forecast API; true historical ERA5 data is not yet fetched.  Backtest
  accuracy is therefore limited by live weather data availability.
