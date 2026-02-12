import pandas as pd
import numpy as np

# Extract


# Time data
def generate_time_data(start_date, end_date, country_holidays):
    time_data = pd.DataFrame({'Timestamp': pd.date_range(start=start_date, end=end_date, freq='h')})
    time_data['Hour'] = time_data['Timestamp'].dt.hour
    time_data['Day'] = time_data['Timestamp'].dt.day
    time_data['Month'] = time_data['Timestamp'].dt.month
    time_data['Year'] = time_data['Timestamp'].dt.year
    time_data['DayOfWeek'] = time_data['Timestamp'].dt.dayofweek
    time_data['DayOfYear'] = time_data['Timestamp'].dt.dayofyear
    time_data['IsWeekend'] = time_data['DayOfWeek'].isin([5, 6]).astype(bool)
    # Get the holidays for the country
    import holidays
    country_holidays = holidays.CountryHoliday(country_holidays)
    time_data['IsHoliday'] = time_data['Timestamp'].dt.date.isin(country_holidays).astype(bool)
    working_days = (time_data['IsWeekend'] == 0) & (time_data['IsHoliday'] == 0)
    time_data['IsWorkingDay'] = working_days.astype(bool)
    return time_data

# Transform

def transform_time_data(time_data):
    # All bool remain as bool, no transformation needed
    # All other columns sin cos transformation
    for col in time_data.columns:
        if col == 'Timestamp' or time_data[col].dtype == 'bool':
            continue
        else:
            max_val = time_data[col].max()
            time_data[f'{col}_sin'] = np.sin(2 * np.pi * time_data[col] / max_val)
            time_data[f'{col}_cos'] = np.cos(2 * np.pi * time_data[col] / max_val)
            time_data.drop(columns=[col], inplace=True)
    
    return time_data

def transform_weather_data(weather_data):
    # Normalize numeric columns except Timestamp
    numeric_cols = weather_data.select_dtypes(include='number').columns
    for col in numeric_cols:
        if col == 'Timestamp':
            continue
        else:
            col_min = weather_data[col].min()
            col_max = weather_data[col].max()
            if col_max != col_min:
                weather_data[col] = (weather_data[col] - col_min) / (col_max - col_min)
            else:
                weather_data[col] = 0.0
    # Make the data hourly by forward filling the values for each hour
    weather_data = weather_data.set_index('Timestamp').resample('h').ffill().reset_index()
    
    return weather_data

def transform_sun_data(sun_data):
    # Transform to hourly data by forward filling the values for each hour
    sun_data = sun_data.set_index('Timestamp').resample('h').ffill().reset_index()
    solar_noon_time = pd.to_datetime(sun_data['Solar_Noon'].astype(str)).dt.time
    solar_noon_dt = pd.to_datetime(sun_data['Timestamp'].dt.date.astype(str) + ' ' + solar_noon_time.astype(str))
    hours = (sun_data['Timestamp'] - solar_noon_dt) / np.timedelta64(1, 'h')
    sun_data["Solar_intensity"] = np.maximum(0, np.cos(hours * np.pi / sun_data['Day_length']))
    sun_data.drop(columns=['Sunrise_time', 'Sunset_time', 'Solar_Noon', 'Day_length'], inplace=True)
    
    return sun_data

def merge_datasets(energy_data, weather_data, sun_data, time_data):
    # Merge on Timestamp
    merged_data = energy_data.merge(weather_data, on='Timestamp', how='left')
    merged_data = merged_data.merge(sun_data, on='Timestamp', how='left')
    merged_data = merged_data.merge(time_data, on='Timestamp', how='left')
    
    return merged_data

def preprocess_merge(energy_data, weather_data, sun_data):
    time_data = generate_time_data(start_date=energy_data['Timestamp'].min(), end_date=energy_data['Timestamp'].max(), country_holidays='UK')
    time_data = transform_time_data(time_data)
    weather_data = transform_weather_data(weather_data) 
    sun_data = transform_sun_data(sun_data) 
    merged_data = merge_datasets(energy_data, weather_data, sun_data, time_data) 
    return merged_data

def main():
    # Energy data
    energy_data = pd.read_excel('Generated_Data_Model.xlsx', sheet_name='Energy_data')
    # Weather data
    weather_data = pd.read_excel('Generated_Data_Model.xlsx', sheet_name='Weather_data_per_zone')
    # Sun data
    sun_data = pd.read_excel('Generated_Data_Model.xlsx', sheet_name='Sun_data')

    energy_data["Timestamp"] = pd.to_datetime(energy_data["Timestamp"])
    weather_data["Timestamp"] = pd.to_datetime(weather_data["Timestamp"])
    sun_data["Timestamp"] = pd.to_datetime(sun_data["Day"])
    sun_data.drop(columns=["Day"], inplace=True)

    expected_columns_energy_train = ['Timestamp', 'Price']
    expected_columns_energy_test = ['Timestamp']
    expected_columns_weather = [
        "MaxTemp",
        "MinTemp",
        "UvIndex",
        "Wind",
        "Dew_point",
        "Cloud_cover",
        "Wind_speed",
        "Pressure",
        "Apparent_temp_max",
        "Apparent_temp_min",
        "Visibility",
        "Humidity",
    ]
    sun_data_columns = ["Sunrise_time", "Sunset_time", "Solar_Noon", "Day_length"]

    # Save expected columns to a .txt file
    with open('expected_columns.txt', 'w') as f:
        f.write("Expected columns for energy_train:\n")
        f.write(", ".join(expected_columns_energy_train) + "\n\n")

        f.write("Expected columns for energy_test:\n")
        f.write(", ".join(expected_columns_energy_test) + "\n\n")

        f.write("Expected columns for weather_data:\n")
        f.write(", ".join(expected_columns_weather) + "\n\n")

        f.write("Expected columns for sun_data:\n")
        f.write(", ".join(sun_data_columns) + "\n")

    with open('expected_columns.txt', 'r') as f:
        contents = f.read()
        expected_columns_energy_train = contents.split("Expected columns for energy_train:\n")[1].split("\n\n")[0].split(", ")
        expected_columns_energy_test = contents.split("Expected columns for energy_test:\n")[1].split("\n\n")[0].split(", ")
        expected_columns_weather = contents.split("Expected columns for weather_data:\n")[1].split("\n\n")[0].split(", ")
        sun_data_columns = contents.split("Expected columns for sun_data:\n")[1].split("\n\n")[0].split(", ")

    merged = preprocess_merge(energy_data, weather_data, sun_data)

    # Save the merged dataset to a new csv file
    merged.to_csv('merged_dataset.csv', index=False)


if __name__ == "__main__":
    main()
