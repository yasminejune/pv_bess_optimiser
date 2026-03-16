# ETL Pipeline: Merged Dataset Construction

ETL script to build a merged dataset from price data, hourly weather, and daily sun data for machine learning model training and analysis.

## Overview

The ETL pipeline combines multiple data sources into a unified hourly dataset with engineered features for time series forecasting. It handles data cleaning, temporal alignment, feature engineering, and quality validation.

## Input Sources

- `Data/price_data.csv`: Electricity price data with temporal resolution
- `Data/historical_hourly_2025.csv`: Weather data at hourly intervals  
- `Data/historical_daily_2025.csv`: Solar and astronomical data at daily resolution

## Outputs

- `merged_dataset.csv`: Hourly merged data with engineered features
- `expected_columns.txt`: Expected column lists used in the validation pipeline

## Key Functions
## Key Functions

### Core Pipeline Functions

#### `preprocess_merge(weather_path, sun_path, price_path, output_path)`
Main orchestration function that coordinates the complete ETL pipeline from raw inputs to final merged dataset.

#### `merge_datasets(weather_data, sun_data, price_data)`
Performs temporal alignment and merging of the three data sources using timestamp-based joins.

#### `transform_price_data(price_data)`
Processes price data for integration, including:
- Temporal resampling to hourly using mean aggregation for numeric columns
- First-value aggregation for non-numeric metadata columns
- Timestamp standardization and validation

#### `transform_weather_data(weather_data)`
Applies weather data preprocessing including:
- Normalization of numeric weather columns using StandardScaler
- Column validation and type checking
- Missing value handling and quality flags

#### `transform_sun_data(sun_data)`
Processes solar and astronomical data with:
- Solar intensity feature computation from multiple radiation components
- Daily to hourly temporal expansion through interpolation
- Solar position and timing feature derivation

#### `generate_time_data(start_time, end_time, freq)`
Creates comprehensive time-based features from timestamp information.

#### `transform_time_data(time_data)`
Engineers temporal features including:
- Cyclical encoding of time components (hour, day, month)
- Weekend/weekday indicators
- Seasonal patterns and calendar effects

### Utility Functions

#### `get_timestamp_range(dataframe)`
Returns the minimum and maximum timestamp values in a DataFrame.

**Parameters:**
- `dataframe`: DataFrame with a `Timestamp` column

**Returns:** 
- Tuple of `(min_timestamp, max_timestamp)` as pandas Timestamps

**Usage:**
```python
start_time, end_time = get_timestamp_range(price_data)
print(f"Data spans from {start_time} to {end_time}")
```

#### `filter_by_timestamp_range(dataframe, start_date, end_date)`
Filter a DataFrame to rows whose timestamp falls within the specified range.

**Parameters:**
- `dataframe`: DataFrame with a `Timestamp` column  
- `start_date`: Inclusive lower bound for timestamp filtering
- `end_date`: Inclusive upper bound for timestamp filtering

**Returns:**
- Filtered copy of the DataFrame within the specified time window

**Usage:**
```python
# Filter data to specific time period
filtered_data = filter_by_timestamp_range(
    weather_data, 
    pd.Timestamp('2025-01-01'), 
    pd.Timestamp('2025-12-31')
)
```

#### `add_lagged_features(merged_data, numeric_columns, lag_periods)`
Generates lag features for time series modeling by creating shifted versions of numeric columns.

**Parameters:**
- `merged_data`: Base dataset for feature engineering
- `numeric_columns`: List of column names to create lag features for
- `lag_periods`: List of integers specifying lag periods (e.g., [1, 2, 24] for 1-hour, 2-hour, 24-hour lags)

### Data Validation Functions

#### `log_dataset_ranges(ranges_by_name)`
Prints timestamp range information for each dataset to facilitate debugging and validation.

#### `standardize_timestamp_column(df, timestamp_col_name)`
Standardizes timestamp column formatting and ensures consistent datetime parsing across all input sources.

## Feature Engineering Details

### Time Features
- **Cyclical Encoding**: Hour, day of week, month encoded as sin/cos pairs to capture cyclical patterns
- **Calendar Effects**: Weekend indicators, seasonal patterns, holiday flags
- **Temporal Trends**: Linear time trends and period-specific effects

### Weather Features  
- **Normalization**: All numeric weather variables standardized using z-score normalization
- **Missing Value Handling**: Forward-fill and interpolation strategies for weather data gaps
- **Derived Variables**: Computed features from raw meteorological measurements

### Solar Features
- **Solar Intensity**: Composite measure derived from multiple solar radiation components
- **Daily Interpolation**: Conversion of daily solar measurements to hourly resolution
- **Solar Position**: Sun angle and position calculations for enhanced solar modeling

### Price Features
- **Temporal Aggregation**: Original price data resampled to hourly resolution
- **Statistical Features**: Rolling means, volatility measures, and trend indicators
- **Lag Features**: Historical price values at configurable lag periods (1h, 2h, 24h, etc.)

## Output Schema

The merged dataset contains the following categories of columns:

### Timestamp Columns
- `Timestamp`: Primary datetime index in standardized format

### Price Columns  
- Original price variable(s) from input price data
- Aggregated price statistics (hourly means, etc.)

### Weather Columns
- Normalized weather variables (temperature, humidity, wind speed, etc.)
- Original weather column names preserved with standardized values

### Solar Columns
- `solar_intensity`: Engineered composite solar radiation measure
- Original solar radiation components from daily data
- Solar position and timing variables

### Time Feature Columns
- Cyclical time encodings (`hour_sin`, `hour_cos`, `day_sin`, `day_cos`, etc.)
- Calendar indicators (`is_weekend`, seasonal flags)
- Time trend variables

### Lag Feature Columns
- Lagged versions of numeric variables with naming pattern: `{original_name}_lag_{period}`
- Configurable lag periods based on pipeline parameters 

## How to Run
## How to Run

### Basic Execution
1. Ensure the required CSV files are present in the `Data/` folder
2. Run the script from the repository root:
   ```bash
   python -m src.ors.etl.etl
   ```

### Command Line Interface
The ETL pipeline includes a command-line interface for customized execution:

```bash
# Basic run with default parameters
python -m src.ors.etl.etl

# Custom file paths and output location  
python -m src.ors.etl.etl \
    --weather-path Data/custom_weather.csv \
    --sun-path Data/custom_sun.csv \
    --price-path Data/custom_prices.csv \
    --output-path custom_output.csv
```

## Pipeline Notes

### Data Processing Details
- **Price Resampling**: Price data resampled to hourly using mean aggregation for numeric columns and first-value selection for metadata
- **Weather Normalization**: All numeric weather columns automatically normalized using StandardScaler
- **Solar Intensity Calculation**: Composite solar feature computed from available radiation components
- **Lag Feature Generation**: Automatic lag feature creation for all numeric columns with configurable periods
- **Temporal Alignment**: All datasets aligned to common hourly timestamp index with consistent timezone handling

### Data Quality Validation
- **Column Validation**: Expected column lists validated against `expected_columns.txt` reference
- **Timestamp Consistency**: Timestamp formats standardized across all input sources
- **Missing Data Handling**: Systematic handling of missing values with quality tracking
- **Range Validation**: Automatic detection and logging of timestamp ranges for each input dataset

### Performance Considerations
- **Memory Efficiency**: Chunked processing for large datasets
- **Computation Optimization**: Vectorized operations for feature engineering
- **I/O Optimization**: Efficient CSV reading/writing with appropriate data types

## Configuration and Customization

### Modifying Input Sources
If dataset names or locations change:
- Update file paths in `main()` function where each CSV is loaded
- Ensure new datasets follow expected schema patterns
- Update `expected_columns.txt` if column structures change

### Adding New Features
If new columns are added to input sources:
- **Weather Data**: New numeric columns will be automatically normalized
- **Weather Data (Non-numeric)**: Add explicit handling in `transform_weather_data()` or exclude from processing  
- **Solar Data**: Update solar intensity calculation in `transform_sun_data()` if new radiation components are available
- **Price Data (Non-numeric)**: Review hourly aggregation strategy in `transform_price_data()` for new metadata columns

### Customizing Feature Engineering
- **Lag Periods**: Modify `lag_periods` parameter in `add_lagged_features()` to change temporal lag features
- **Time Features**: Extend `transform_time_data()` for additional temporal patterns (holidays, business cycles, etc.)
- **Normalization Strategy**: Replace StandardScaler in `transform_weather_data()` for alternative scaling methods

## Error Handling and Debugging

### Common Issues
- **Missing Files**: Verify all input CSV files exist in expected locations
- **Column Mismatches**: Check that input data follows expected schema in `expected_columns.txt`
- **Timestamp Parsing**: Ensure timestamp columns use consistent datetime formats
- **Memory Limits**: For large datasets, consider chunked processing or increased memory allocation

### Debugging Tools
- **Range Logging**: `log_dataset_ranges()` provides timestamp coverage information
- **Column Validation**: Compare actual vs. expected columns using generated `expected_columns.txt`
- **Intermediate Outputs**: Enable intermediate CSV saves for step-by-step validation

### Quality Assurance
- **Output Validation**: Verify merged dataset completeness and feature coverage
- **Statistical Checks**: Review feature distributions and identify anomalies
- **Temporal Consistency**: Confirm proper timestamp alignment across merged sources