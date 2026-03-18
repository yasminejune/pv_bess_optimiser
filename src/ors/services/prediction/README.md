# Prediction Service

This is the canonical prediction package for the project.

- Python package path: `ors.services.prediction`
- Source location: `src/ors/services/prediction/`
- Model: LightGBM `ForecasterRecursive` (skforecast), trained on 15-min UK electricity price data.

## Modules

- `data_pipeline.py`:
  - `load_source_data(project_root)` — loads and validates the three source CSVs.
  - `build_merged_dataset(project_root)` — full notebook-aligned preprocessing pipeline:
    reindexes price to 15-min, interpolates hourly weather, forward-fills daily weather,
    computes `Solar_intensity`, patches price outliers (±2 std), and engineers
    time/cyclic/calendar features.
  - `get_feature_cols(df)` — returns the ordered feature column list from a preprocessed DataFrame.
- `prediction_model.py`:
  - feature prep (`resolve_target_column`, `prepare_features`, `prepare_features_for_inference`)
  - split/train/evaluate (`time_based_split`, `train_xgb_regressor`, `evaluate_model`) — XGBoost legacy
  - LightGBM helpers (`_build_lgbm_recursive`, `_train_and_save_lgbm`)
  - persistence (`save_metrics`, `save_predictions`, `save_feature_importance`)
  - orchestration (`train_and_save_from_dataframe`, `train_and_save`, `main`)
    — dispatches to LightGBM when `model_name` contains `"lgbm"` (default).
- `report_generator.py`:
  - loads saved artifacts and produces PDF reports.
  - auto-detects LightGBM vs XGBoost model type and shows appropriate summary stats.
  - includes plots for actual vs predicted, residuals, and feature importance.
- `hyperparameter_search.py`:
  - grid-search style tuning and best-model artifact export.
- `train_script.py`:
  - one-command train + report pipeline (default model: `lgbm_recursive`).
- `run_sample.py`:
  - deterministic sample run from a fixed merged dataset.

## CLI Usage

Run from repo root using the canonical package path:

- Train only:
  - `python -m ors.services.prediction.prediction_model`
- Train + report pipeline:
  - `python -m ors.services.prediction.train_script`
  - `python -m ors.services.prediction.train_script --model-name lgbm_recursive`
- Hyperparameter search:
  - `python -m ors.services.prediction.hyperparameter_search`
- Report generation:
  - `python -m ors.services.prediction.report_generator`
- Sample run:
  - `python -m ors.services.prediction.run_sample`

## Input Data

Expected files in `Data/`:

- `price_data_rotated_2d.csv` — 15-min electricity price and demand data (2023–present).
- `historical_hourly_2023_2025.csv` — hourly weather observations.
- `historical_daily_2023_2025.csv` — daily weather observations (includes sunrise/daylight_duration for Solar_intensity).

## Output Artifacts

Training runs save artifacts under:

- `Prediction/Models/<model_name>_<timestamp>/model.joblib` — fitted `ForecasterRecursive`
- `Prediction/Models/<model_name>_<timestamp>/model_meta.joblib` — metadata dict (feature_cols, horizon, max_lag, etc.)
- `Prediction/Models/<model_name>_<timestamp>/metrics.json` — test-split MAE / RMSE / MAPE
- `Prediction/Models/<model_name>_<timestamp>/predictions.csv` — test-split actual vs predicted
- `Prediction/Models/<model_name>_<timestamp>/feature_importance.csv`
- `Prediction/Models/<model_name>_<timestamp>/metadata.json`
- `Prediction/Models/<model_name>_<timestamp>/model_report.pdf`

An inference copy is also written to:

- `models/price_prediction/lgbm_recursive_single_model/recursive_single_model_<timestamp>.joblib`
- `models/price_prediction/lgbm_recursive_single_model/recursive_single_model_<timestamp>_meta.joblib`

## Model Configuration

| Parameter | Value |
|---|---|
| Algorithm | LightGBM `ForecasterRecursive` (skforecast) |
| Objective | MAE |
| n_estimators | 850 |
| learning_rate | 0.025 |
| num_leaves | 127 |
| max_depth | 11 |
| Lags | 1–48 + [64, 96, 128, 192, 288, 384, 480, 576, 672] |
| Rolling features | mean/std/min/max over windows [4, 16, 32, 96, 192, 288, 384, 576] |
| Training window | Most recent 180 days |
| Horizon | 96 steps (24 h at 15-min resolution) |
| Validation split | 21 days (2 016 steps) |
| Test split | 21 days (2 016 steps) |
| random_state | 42 |

## Preprocessing Notes

- **Price reindexing**: full 15-min index filled from price CSV min/max timestamps.
- **Outlier patching**: prices outside ±2 std of the global mean are replaced with the
  time-of-day slot median computed from in-range values.
- **Weather interpolation**: hourly weather columns are interpolated to 15-min by time;
  `weather_code` is forward-filled. Daily features are forward-filled to 15-min.
- **Solar_intensity**: cosine of hours-from-solar-noon normalised by day length.
- **Feature set**: 42 columns — hourly weather (16), daily weather (7), demand/generation (7),
  cyclic time encodings (4), calendar categoricals (7), `is_price_patched` dropped from
  exog but kept in raw df.
- **Split strategy**: fixed last-N-steps splits (no fraction-based split); val and test are
  each the final 21 days of the dataset.
