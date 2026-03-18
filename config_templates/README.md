# Configuration Templates for Optimization

This directory contains template JSON configuration files that clients can fill out to run optimizations without any code changes.

## Quick Start

1. **Copy a template**: Choose the template that best matches your setup
2. **Fill in your values**: Replace placeholder values with your system specifications
3. **Run optimization**: Use the main optimization script with your config file

```bash
python run_optimization.py your_config.json
```

## Available Templates

### 📋 `example_full_config.json`
**Complete example showing all available options**
- PV system with weather-based forecasting
- Battery system with all parameters
- Forecasted price data
- Detailed output options
- **Use this as reference for all available settings**

### ⚡ `battery_only_simple.json` 
**Minimal battery-only configuration**
- Battery energy storage only (no PV)
- Essential parameters only
- Automatic price forecasting
- **Best for: Simple battery optimization**

### 📊 `manual_data_config.json`
**Custom data input configuration**
- Manual price profile (hourly prices)
- Manual PV generation profile (hourly generation)
- **Best for: Testing specific scenarios**

## Configuration Sections

### 🏷️ **Metadata**
```json
{
  "config_name": "Your Project Name",
  "created_by": "Your Name", 
  "notes": "Description of this optimization run"
}
```

### ☀️ **PV System** (Optional)
```json
{
  "pv": {
    "rated_power_kw": 1000.0,           // Peak power capacity
    "max_export_kw": 800.0,             // Grid export limit
    "generation_source": "forecast",     // "forecast", "historical", "manual"
    "location_lat": 51.5074,            // For weather forecasting
    "location_lon": -0.1278
  }
}
```

**Generation Sources:**
- `"forecast"`: Use weather forecasting (requires lat/lon)
- `"historical"`: Use historical data file (requires file path)
- `"manual"`: Use provided hourly profile

### 🔋 **Battery System** (Required)
```json
{
  "battery": {
    "rated_power_mw": 100.0,           // Max charge/discharge rate
    "energy_capacity_mwh": 600.0,      // Total capacity
    "current_energy_mwh": 300.0,       // Current stored energy
    "current_mode": "idle",            // "idle", "charging", "discharging"
    "cycles_used_today": 0             // Already used cycles
  }
}
```

**Current State Options:**
- Provide either `current_energy_mwh` OR `current_soc_percent`
- Set `current_mode` to actual battery operation
- Track `cycles_used_today` if continuing mid-day

### 🕐 **Optimization Period**
```json
{
  "optimization": {
    "optimization_date": "2026-03-08",  // YYYY-MM-DD format
    "start_time": "00:00",              // HH:MM format
    "duration_hours": 24,               // 1-48 hours
    "price_source": "forecast"          // "forecast", "historical", "manual"
  }
}
```

### 📁 **Output Settings**
```json
{
  "output": {
    "output_csv_path": "results.csv",
    "include_summary": true,
    "include_recommendations": true,
    "verbose": false
  }
}
```

## Data Source Options

### 🔮 **Automatic Forecasting** (Recommended)
- **Price**: `"price_source": "forecast"`
- **PV**: `"generation_source": "forecast"` (requires location)
- System automatically gets latest forecasts

### 📈 **Historical Data**
- **Price**: Provide `historical_price_path` to CSV file
- **PV**: Provide `historical_data_path` to CSV file
- Good for backtesting or when forecasts unavailable

### ✍️ **Manual Data Entry**
- **Price**: Provide `manual_price_profile` with hourly prices
- **PV**: Provide `manual_generation_profile` with hourly generation
- Perfect for testing specific scenarios

## Common Use Cases

### 🏠 **Residential Setup**
```json
{
  "pv": { "rated_power_kw": 10.0 },
  "battery": { 
    "rated_power_mw": 0.01, 
    "energy_capacity_mwh": 0.02 
  }
}
```

### 🏭 **Commercial Setup**
```json
{
  "pv": { "rated_power_kw": 500.0 },
  "battery": { 
    "rated_power_mw": 2.0, 
    "energy_capacity_mwh": 6.0 
  }
}
```

### ⚡ **Utility Scale**
```json
{
  "pv": { "rated_power_kw": 50000.0 },
  "battery": { 
    "rated_power_mw": 100.0, 
    "energy_capacity_mwh": 600.0 
  }
}
```

## Validation

The system automatically validates your configuration:
- ✅ **Parameter ranges**: Power/energy values within realistic bounds
- ✅ **Consistency**: SOC percentages match energy values
- ✅ **Completeness**: Required fields are provided
- ✅ **Date/time format**: Proper formatting for dates and times

## Getting Help

- **Configuration errors**: Check the validation messages for specific issues
- **Data problems**: Ensure file paths exist and data formats are correct
- **Results questions**: Check the output summary and recommendations

**Need help?** Contact the optimization team with your configuration file and any error messages.