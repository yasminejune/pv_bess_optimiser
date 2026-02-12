ETL script to build a merged dataset from three sources: energy, weather, and sun.

Inputs
- Generated_Data_Model.xlsx with sheets:
	- Energy_data
	- Weather_data_per_zone
	- Sun_data

Outputs
- merged_dataset.csv (hourly merged data)
- expected_columns.txt (expected column lists used in the pipeline)

How to run
1. Place Generated_Data_Model.xlsx in the same folder as ETL.py.
2. Run the script:
	 python ETL.py

Notes
- The script creates time features, normalizes numeric weather columns, and computes a solar intensity feature.
- Timestamp columns are required in all three source sheets.

If the dataset name changes
- Update the file name in `main()` where the Excel file is loaded.
- If the sheet names change, update the `sheet_name` values for energy, weather, and sun reads.

If new columns are added
- Add or update the expected column lists so `expected_columns.txt` stays accurate.
- If new weather columns are numeric, they will be normalized automatically.
- If new weather columns are non-numeric, either exclude them or update the transform to handle them explicitly.
- If new sun columns affect solar intensity, adjust the calculation in `transform_sun_data()`.