# Prediction folder overview

This folder contains the training pipeline, visualizations, and PDF reporting utilities for the price prediction workflow. It assumes the ETL pipeline builds a merged dataset from the Excel source in Data/.

## Files and purpose
- [Prediction/prediction_model.py](Prediction/prediction_model.py): Train XGBoost, evaluate metrics, and save artifacts.
- [Prediction/train_script.py](Prediction/train_script.py): One-click training + PDF report pipeline.
- [Prediction/run_sample.py](Prediction/run_sample.py): Deterministic sample run using a fixed CSV.
- [Prediction/visualize_predictions.ipynb](Prediction/visualize_predictions.ipynb): Visual analysis of predictions vs actuals, residuals, and feature importance.
- [Prediction/report_generator.py](Prediction/report_generator.py): Generate a PDF report with metrics, features, and plots.
- [Prediction/Models](Prediction/Models): Per-run model artifacts organized by model name and timestamp.
- [Prediction/Visualization](Prediction/Visualization): Saved figures from the notebook.

## prediction_model.py functions
- import_etl_module(project_root): Adds ETL/ to sys.path and imports ETL as a module.
- load_source_data(project_root): Loads Energy_data, Weather_data, and Daily_weather from Data/Generated_Data_Model.xlsx and parses timestamps.
- build_merged_dataset(project_root): Runs ETL.preprocess_merge and returns a sorted merged dataset.
- prepare_features(df, target_col): Drops target and Timestamp, keeps numeric/bool features, casts bool to int, fills missing values with median.
- time_based_split(features, target, test_size): Splits data in time order to avoid leakage.
- train_xgb_regressor(x_train, y_train, random_state): Trains an XGBoost regressor with fixed hyperparameters.
- evaluate_model(model, x_test, y_test): Computes MAE, RMSE, and R2.
- save_metrics(metrics, output_path): Writes metrics JSON.
- save_predictions(timestamps, y_true, y_pred, output_path): Writes predictions CSV.
- save_feature_importance(model, features, output_path): Saves feature importances to CSV.
- train_and_save(project_root, model_name, test_size): Full training pipeline that returns the run folder.
- train_and_save_from_dataframe(project_root, df, model_name, test_size): Same pipeline but uses a fixed dataframe (sample or custom).
- prepare_features_for_inference(df, target_col): Feature prep for prediction inputs.
- predict_prices(model, df, target_col): Predicts using the inference feature pipeline.
- main(): Orchestrates the full training run and writes outputs in Prediction/.

Outputs from prediction_model.py
- Prediction/Models/<model>_<YYYYMMDD_HHMMSS>/model.json
- Prediction/Models/<model>_<YYYYMMDD_HHMMSS>/metrics.json
- Prediction/Models/<model>_<YYYYMMDD_HHMMSS>/predictions.csv
- Prediction/Models/<model>_<YYYYMMDD_HHMMSS>/feature_importance.csv
- Prediction/Models/<model>_<YYYYMMDD_HHMMSS>/metadata.json

## visualize_predictions.ipynb sections
1. Load Artifacts and Source Data: Loads predictions/metrics and rebuilds merged data via ETL.
2. Rebuild Features and Time-Based Split: Recreates the feature pipeline and split to align with saved predictions.
3. Join Predictions with Timestamps: Merges predictions with actuals and computes residuals.
4. Plot Actual vs Predicted Over Time: Time series comparison plot.
5. Plot Residuals and Error Distribution: Residuals over time and histogram.
6. Plot Feature Importance: Horizontal bar chart of top features.
7. Save Figures to Disk: Writes plots to Prediction/Visualization.

## report_generator.py functions
- find_project_root(start): Finds the project root by looking for ETL/.
- import_etl_module(project_root): Imports the ETL module.
- load_source_data(project_root): Loads input Excel sheets and parses timestamps.
- build_merged_dataset(project_root): Builds merged dataset using ETL.preprocess_merge.
- prepare_features(df, target_col): Same feature preparation as training.
- time_based_split(features, target, test_size): Time-based split.
- load_metrics(metrics_path): Loads metrics JSON into a dict.
- load_feature_importance(path): Loads feature importance CSV.
- load_model_metadata(path): Loads optional model metadata JSON.
- create_text_page(pdf, title, lines): Adds a text page to the PDF.
- create_table_page(pdf, title, df): Adds a table page to the PDF.
- plot_actual_vs_predicted(df): Plot of actual vs predicted prices.
- plot_residuals(df): Residuals over time.
- plot_residual_distribution(df): Residual histogram.
- plot_feature_importance(df, top_n): Feature importance plot.
- build_report(...): Orchestrates data load, alignment, and PDF generation.
- parse_args(): CLI argument parsing.
- main(): CLI entry point; writes Prediction/model_report.pdf by default.

## What to modify when data or variables change
- New input columns in ETL output:
  - Update ETL transformations if new columns are non-numeric or need custom handling.
  - If new columns are numeric, the current pipeline will include them automatically.
- New target column name:
  - Update target_col in prediction_model.py and report_generator.py or pass a new name if you add a parameter.
- Timestamp changes:
  - Ensure Timestamp exists and is parseable in all input sheets and merged output.
- New data source file or sheet names:
  - Update load_source_data in prediction_model.py and report_generator.py.
- Custom preprocessing or encodings:
  - Adjust prepare_features and update the report encoding notes in report_generator.py.

## How to run
- Training:
  - python Prediction/prediction_model.py
- Full pipeline (train + report):
  - python Prediction/train_script.py
- Deterministic sample run (fixed dataset):
  - python Prediction/run_sample.py
- Visuals:
  - Open Prediction/visualize_predictions.ipynb and run all cells
- PDF report:
  - python Prediction/report_generator.py
  - Optional args: --run-dir, --preds, --metrics, --importance, --model-name, --metadata, --test-size

## Tests
- Run all tests from the repo root:
  - python -m unittest discover -s Prediction/tests -p "test_*.py"

## Fixed inputs for reproducibility
- Sample dataset: [Prediction/sample_data/merged_sample.csv](Prediction/sample_data/merged_sample.csv)
- The sample run writes outputs to Prediction/Models with a model name of xgboost_sample.

## Generalizing to other models
- Save your model outputs into the same CSV/JSON formats:
  - Predictions CSV: Timestamp, Price_pred, and optionally Price_true
  - Metrics JSON: numeric metrics in a flat dict
  - Feature importance CSV: feature, importance
- Run the report generator with model-specific inputs and metadata:
  - python Prediction/report_generator.py --run-dir PATH --model-name "YourModel"

## Switching to a different model (TabNet, neural network, etc.)
If you change the model inside [Prediction/prediction_model.py](Prediction/prediction_model.py), update the following parts so the rest of the pipeline (reports, notebook, train_script) keeps working:

- train_xgb_regressor: Replace this with your model training function (e.g., train_tabnet, train_nn). Keep the signature similar if possible.
- evaluate_model: Keep returning a dict of numeric metrics (MAE, RMSE, R2), or add additional keys. The report will show any metrics present.
- save_feature_importance: For models without feature importances, either skip it or write a proxy (e.g., permutation importance). The report generator expects a CSV with feature and importance columns.
- train_and_save: Keep writing the same artifact filenames in the run folder:
  - model.json (or another model file extension, but keep the name consistent)
  - metrics.json
  - predictions.csv
  - feature_importance.csv (or an empty file with headers if not available)
  - metadata.json

Notes
- If your model needs different inputs (categorical encodings, scaling, embeddings), update prepare_features accordingly and also update the encoding notes in report_generator.py.
- If your model is probabilistic or outputs intervals, you can add extra columns to predictions.csv (e.g., Price_pred_low, Price_pred_high) and extend the report plots if needed.
