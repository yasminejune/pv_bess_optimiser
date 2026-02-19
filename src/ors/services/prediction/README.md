# Prediction Service

This is the canonical prediction package for the project.

- Python package path: `ors.services.prediction`
- Source location: `src/ors/services/prediction/`
- It depends on ETL outputs and source CSVs under `Data/`.

## Modules

- `data_pipeline.py`:
  - `load_source_data(project_root)` loads and validates source CSVs.
  - `build_merged_dataset(project_root)` builds the merged, sorted dataframe.
- `prediction_model.py`:
  - feature prep (`resolve_target_column`, `prepare_features`, `prepare_features_for_inference`)
  - split/train/evaluate (`time_based_split`, `train_xgb_regressor`, `evaluate_model`)
  - persistence (`save_metrics`, `save_predictions`, `save_feature_importance`)
  - orchestration (`train_and_save_from_dataframe`, `train_and_save`, `main`)
- `report_generator.py`:
  - loads saved artifacts and produces PDF reports
  - includes plots for predictions, residuals, and feature importance
- `hyperparameter_search.py`:
  - grid-search style tuning and best-model artifact export
- `train_script.py`:
  - one-command train + report pipeline
- `run_sample.py`:
  - deterministic sample run from a fixed merged dataset

## CLI Usage

Run from repo root using the canonical package path:

- Train only:
  - `python -m ors.services.prediction.prediction_model`
- Train + report pipeline:
  - `python -m ors.services.prediction.train_script`
- Hyperparameter search:
  - `python -m ors.services.prediction.hyperparameter_search`
- Report generation:
  - `python -m ors.services.prediction.report_generator`
- Sample run:
  - `python -m ors.services.prediction.run_sample`

## Input Data

Expected files in `Data/`:

- `price_data.csv`
- `historical_hourly_2025.csv`
- `historical_daily_2025.csv`

## Output Artifacts

Training/search runs save artifacts under:

- `Prediction/Models/<model_name>_<timestamp>/model.json`
- `Prediction/Models/<model_name>_<timestamp>/metrics.json`
- `Prediction/Models/<model_name>_<timestamp>/predictions.csv`
- `Prediction/Models/<model_name>_<timestamp>/feature_importance.csv`
- `Prediction/Models/<model_name>_<timestamp>/metadata.json`

PDF report default outputs:

- run-specific: `Prediction/Models/<run>/model_report.pdf`
- fallback: `Prediction/model_report.pdf`

## Notes

- `Timestamp` is dropped from model features.
- Boolean features are cast to `0/1`.
- Missing feature values are median-imputed.
- Time-based split is used to avoid leakage.
