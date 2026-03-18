# Prediction Service — Session Change Log

Branch: `regressor_pred`

The goal of this session was to align the training script stack with the
LightGBM notebook (`xgboost_price_forecast_direct.ipynb`), which had been
developed and validated interactively but was not yet wired into the
production training/reporting pipeline.

---

## Files changed

### `data_pipeline.py` — **complete rewrite**

**Before:** called `ors.etl.etl.preprocess_merge` on three old-named CSVs
(`price_data.csv`, `historical_hourly_2025.csv`, `historical_daily_2025.csv`)
and returned a plain merged DataFrame with minimal preprocessing.

**After:** implements the full notebook preprocessing pipeline. Public API is
unchanged (`build_merged_dataset(project_root) -> pd.DataFrame`), but
internally it now:

1. Loads `price_data_rotated_2d.csv`, `historical_hourly_2023_2025.csv`,
   `historical_daily_2023_2025.csv`.
2. Reindexes price series to a continuous 15-min frequency.
3. Interpolates hourly weather to 15-min by time (numeric cols); forward-fills
   `weather_code`.
4. Forward-fills daily weather to 15-min; computes `Solar_intensity` from
   sunrise + daylight duration using the same cosine formula as the notebook.
5. Fills remaining NaNs; drops rows without a valid price.
6. Patches price outliers (±2 std) with the time-of-day slot median computed
   from in-range values only.
7. Engineers `hour`, `quarter_hour`, `quarter_of_day`, `day_of_week`,
   `is_weekend`, `is_holiday` (UK), `is_working_day`, `tod_sin/cos`,
   `dow_sin/cos`, `time_idx`.
8. Casts all feature columns to `float`.

A new helper was also added:

```python
get_feature_cols(df: pd.DataFrame) -> list[str]
```

Returns every column except `Timestamp`, `price`, `price_raw`,
`is_price_patched`, `time_idx` — the same 42-column list the notebook uses as
`feature_cols` / `exog`.

---

### `prediction_model.py` — **additive changes**

The existing XGBoost functions (`prepare_features`, `train_xgb_regressor`,
`evaluate_model`, etc.) are **untouched** — they are still used by the
XGBoost path and by `report_generator.py`.

**Added:**

- `_build_lgbm_recursive(n_estimators) -> ForecasterRecursive`
  Builds `ForecasterRecursive(LGBMRegressor(…))` with the exact notebook
  hyperparameters:

  | param | value |
  |---|---|
  | objective | mae |
  | n_estimators | 850 |
  | learning_rate | 0.025 |
  | num_leaves | 127 |
  | max_depth | 11 |
  | min_child_samples | 20 |
  | subsample / colsample_bytree | 0.95 |
  | reg_alpha / reg_lambda | 0.1 / 0.8 |
  | max_bin | 511 |
  | lags | 1–48 + [64,96,128,192,288,384,480,576,672] |
  | RollingFeatures | mean/std/min/max × windows [4,16,32,96,192,288,384,576] |

- `_train_and_save_lgbm(project_root, df, model_name) -> Path`
  Replicates the notebook training loop:
  - Training window: most recent 180 days (17 280 steps).
  - Fixed val split: 21 days (2 016 steps); fixed test split: 21 days.
  - Evaluates the already-fitted model on the test split using the same
    rolling-origin window approach as the notebook.
  - Saves artifacts to `Prediction/Models/<model_name>_<timestamp>/`:
    - `model.joblib` — fitted `ForecasterRecursive`
    - `model_meta.joblib` — full metadata dict including `feature_cols`
    - `metrics.json` — `mae`, `rmse`, `mape` on test split
    - `predictions.csv` — `Timestamp`, `Price_true`, `Price_pred`
    - `feature_importance.csv` — from `model.get_feature_importances()`
    - `metadata.json` — scalar metadata for the report
  - Also writes an inference copy to
    `models/price_prediction/lgbm_recursive_single_model/`.

**Updated:**

- `train_and_save_from_dataframe` — default `model_name` changed from
  `"xgboost"` to `"lgbm_recursive"`. A dispatch at the top of the function
  body routes to `_train_and_save_lgbm` when `"lgbm" in model_name.lower()`;
  otherwise falls through to the existing XGBoost path unchanged.

- Top-level import: `get_feature_cols` added to the import from
  `.data_pipeline`.

---

### `train_script.py` — **minimal changes**

- `--model-name` default changed from `"xgboost"` to `"lgbm_recursive"`.
- Module docstring updated.

Everything else (argument parsing, `build_merged_dataset` call,
`train_and_save_from_dataframe` call, `build_report` call) is unchanged.

---

### `report_generator.py` — **LGBM/XGBoost branching**

**Before:** always ran `prepare_features` → `time_based_split` (XGBoost
assumptions) to reconstruct the test split, and used XGBoost-specific
summary text.

**After:** detects model type via
`is_lgbm = "lgbm" in model_name.lower() or "horizon" in metadata`
and branches:

**LGBM path:**
- Skips `prepare_features` / `time_based_split` entirely.
- Feature list from `get_feature_cols(merged_df)`.
- `predictions.csv` already contains `Price_true` / `Price_pred` — used
  directly without any merge.
- Report summary page shows: row count, feature count, n_estimators, horizon,
  max_lag, train_end_idx, val/test step counts.
- Preprocessing notes page describes the LightGBM pipeline.

**XGBoost path:** unchanged from before.

**Bug fixed:** `features_df` previously referenced `features.columns` /
`features.dtypes` which was only defined in the XGBoost branch. Now uses
`feature_list` (defined in both branches) and reads dtypes from `merged_df`.

Additional import: `get_feature_cols` added from `.data_pipeline`.

---

### `README.md` (`src/ors/services/prediction/`) — **full rewrite**

Updated to reflect:
- Correct input file names.
- `get_feature_cols` in `data_pipeline` description.
- `_build_lgbm_recursive` / `_train_and_save_lgbm` under `prediction_model`.
- XGBoost described as legacy; LightGBM as default.
- `report_generator` auto-detection behaviour documented.
- Output artifacts table (`.joblib` files, inference path).
- Model configuration table (all hyperparameters).
- Preprocessing notes section.

---

### `pyproject.toml` — **dependency additions**

```toml
"lightgbm>=4.0.0",
"skforecast==0.20.1",
```

Added to `[project.dependencies]`.

```toml
"lightgbm.*",
"skforecast.*",
"joblib.*",
```

Added to `[[tool.mypy.overrides]]` so mypy ignores missing stubs for these
packages.

---

## How to run

```bash
# 1. Install (picks up new deps)
pip install -e .

# 2. Train + generate PDF report
python -m ors.services.prediction.train_script --model-name lgbm_recursive

# If ors is not found after install:
set PYTHONPATH=src
python -m ors.services.prediction.train_script --model-name lgbm_recursive
```

Outputs land in `Prediction/Models/lgbm_recursive_<timestamp>/` and the
inference joblib in `models/price_prediction/lgbm_recursive_single_model/`.

---

## Known issues / next steps

- `report_generator.py` still imports `resolve_target_column`,
  `prepare_features`, `time_based_split` from `prediction_model` (used only
  for the XGBoost legacy path). These imports will cause a `mypy` warning if
  the XGBoost path is ever removed.
- The `test_size` argument in `train_and_save_from_dataframe` is silently
  ignored for LGBM models (the split is always fixed at 21 days). This is
  intentional but could be confusing if called directly.
- `run_sample.py` and `hyperparameter_search.py` have not been reviewed for
  compatibility with the new LGBM pipeline.
