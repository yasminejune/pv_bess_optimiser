ETL script to build a merged dataset from price data, hourly weather, and daily sun data.

Inputs
- Data/price_data.csv
- Data/historical_hourly_2025.csv
- Data/historical_daily_2025.csv

Outputs
- merged_dataset.csv (hourly merged data)
- expected_columns.txt (expected column lists used in the pipeline)

How to run
1. Ensure the CSV files are present in the Data folder.
2. Run the script from the repo root:
	 python -m ETL.ETL

Notes
- The script creates time features, normalizes numeric weather columns, and computes a solar intensity feature.
- Price data is resampled to hourly using mean for numeric columns and first for non-numeric columns.
- Timestamp columns are required in all three source files.

If the dataset name changes
- Update the file name in `main()` where each CSV is loaded.

If new columns are added
- Add or update the expected column lists so `expected_columns.txt` stays accurate.
- If new weather columns are numeric, they will be normalized automatically.
- If new weather columns are non-numeric, either exclude them or update the transform to handle them explicitly.
- If new sun columns affect solar intensity, adjust the calculation in `transform_sun_data()`.
- If new price columns are non-numeric, review the hourly aggregation in `transform_price_data()`.