# Battery Energy Storage Optimization - User Guide

🚀 **Transform your battery operations with AI-powered optimization**

This system provides optimal charge/discharge schedules for battery energy storage systems (BESS) with optional solar PV integration. No code changes required - just configure and run!

## 🎯 Quick Start (3 Steps)

### 1. **Choose Your Template**
Copy a configuration template that matches your setup:

```bash
cp config_templates/residential_home.json my_config.json          # Home battery
cp config_templates/commercial_warehouse.json my_config.json      # Commercial 
cp config_templates/utility_scale_grid.json my_config.json       # Utility scale
```

### 2. **Fill In Your Details**
Edit `my_config.json` with your system specifications:

- **Battery**: Power rating, capacity, current state
- **Solar** (optional): Panel size, location
- **Date**: When to optimize (any date)
- **Output**: Where to save results

### 3. **Run Optimization**
```bash
python run_optimization.py my_config.json
```

**Output**: CSV file with optimal actions for every 15 minutes!

---

## 📊 What You Get

### **Main Results CSV**
Timestamp-by-timestamp recommendations:

| Timestamp | Action | Power (MW) | Battery SOC | Profit (£) | Price (£/MWh) |
|-----------|--------|------------|-------------|------------|---------------|
| 00:00 | CHARGE from grid | 50.0 | 45% | -2.50 | 25.00 |
| 18:00 | DISCHARGE to grid | 85.0 | 65% | +8.50 | 85.00 |

### **Summary Report**  
- 💰 **Total profit** for the optimization period
- ⚡ **Energy utilization** and cycle efficiency
- 💡 **Actionable recommendations** for operations
- ⏰ **Time range**: "These are the optimal choices from XX:XX to XX:XX"

---

## ⚙️ Configuration Options

### 🔋 **Battery Configuration**
```json
{
  "battery": {
    "rated_power_mw": 100.0,        // Max charge/discharge rate
    "energy_capacity_mwh": 600.0,   // Total capacity
    "current_soc_percent": 50.0,    // Current charge level
    "current_mode": "idle",         // Current operation
    "cycles_used_today": 0          // Already used cycles
  }
}
```

**Key Points:**
- ✅ **Real battery state**: Uses actual SOC, not hardcoded 50%
- ✅ **Current operations**: Continues from charging/discharging state
- ✅ **Daily limits**: Tracks cycles already used today

### ☀️ **Solar PV Configuration** (Optional)
```json
{
  "pv": {
    "rated_power_kw": 500.0,        // Peak solar capacity
    "location_lat": 51.5074,        // For weather forecasts
    "location_lon": -0.1278,        // London coordinates
    "generation_source": "forecast"  // Auto weather forecasting
  }
}
```

**Generation Sources:**
- 🔮 **"forecast"**: Automatic weather-based forecasting
- 📈 **"historical"**: Use your historical data files
- ✍️ **"manual"**: Provide hourly generation profile

### 📅 **Optimization Period**
```json
{
  "optimization": {
    "optimization_date": "2026-03-08",  // Any date
    "start_time": "00:00",              // Start time
    "duration_hours": 24,               // How long to optimize
    "time_step_minutes": 15             // Resolution (15/30/60 min)
  }
}
```

---

## 🏠 Common Use Cases

### **Residential Home**
- **Battery**: 5-15 kW / 10-30 kWh
- **Goal**: Reduce electricity bills
- **Features**: Solar self-consumption, peak avoiding

### **Commercial Building**  
- **Battery**: 100-500 kW / 200-1000 kWh
- **Goal**: Demand charge reduction
- **Features**: Peak shaving, load shifting

### **Utility Scale**
- **Battery**: 10-200 MW / 40-800 MWh  
- **Goal**: Grid services and arbitrage
- **Features**: Frequency regulation, energy arbitrage

### **Microgrids**
- **Battery**: Various sizes
- **Goal**: Energy independence
- **Features**: Backup power, renewable integration

---

## 🔄 Data Sources

### **Automatic (Recommended)**
✅ Latest price forecasts  
✅ Weather-based solar forecasts  
✅ No data files needed  

Just provide location coordinates!

### **Historical Data**
📁 Use your existing CSV files:  
- Price data: `timestamp, price`  
- Solar data: `timestamp, generation_kw`  

### **Manual Profiles**
✍️ Define hourly patterns:
```json
{
  "manual_price_profile": {
    "08:00": 45.0,
    "18:00": 85.0,
    "22:00": 35.0
  }
}
```

---

## 💡 Pro Tips

### **⚡ Maximize Profits**
1. **Check cycle limits**: More cycles = more arbitrage opportunities
2. **Monitor price spreads**: 50+ £/MWh spreads are profitable
3. **Use solar storage**: Store cheap solar for expensive periods

### **🔧 Fine-Tuning**
1. **Adjust SOC limits**: Wider range = more flexibility
2. **Set realistic efficiency**: Account for real-world losses
3. **Update current state**: Always use actual battery status

### **📊 Interpret Results**
1. **Focus on profit/hour**: Shows sustainable performance
2. **Check energy utilization**: >70% is good efficiency
3. **Review recommendations**: Specific operational guidance

---

## ❓ Troubleshooting

### **Configuration Errors**
```
❌ ERROR: current_energy_mwh must be between 60.0 and 540.0 MWh
```
**Fix**: Adjust `current_energy_mwh` to be within SOC limits

### **Data Loading Issues**  
```
⚠️ Price loading failed: File not found
```
**Fix**: Check file paths or switch to `"price_source": "forecast"`

### **Solver Problems**
```
❌ No optimization solver available
```
**Fix**: Install a solver:
```bash
pip install highspy  # Recommended
# OR
conda install glpk
```

### **Validation Only**
Test your configuration without running:
```bash
python run_optimization.py my_config.json --validate
```

---

## 🎯 Success Examples

### **Example 1: Peak Shaving Success**
```
✅ OPTIMIZATION COMPLETED SUCCESSFULLY!
Period: 2026-03-08 from 00:00 to 24:00  
Total Profit: £1,247.50
Cycles Used: 3/3
Energy Utilization: 85.2%

💡 RECOMMENDATIONS:
• Average charging price: £32.50/MWh
• Average discharge price: £78.20/MWh  
• Stored 145.2 MWh of solar generation for later discharge

🎯 These are the optimal choices from 2026-03-08 00:00 to 24:00
```

### **Example 2: Solar + Storage Optimization**
```
✅ OPTIMIZATION COMPLETED SUCCESSFULLY!
Period: 2026-03-08 from 06:00 to 20:00
Total Profit: £3,456.78 
Cycles Used: 2/4
Energy Utilization: 76.3%

💡 RECOMMENDATIONS:
• Could utilize 2 additional cycles for more arbitrage
• Stored 78% of solar generation for later discharge
• Peak solar storage at 12:30, discharge at 19:00

🎯 These are the optimal choices from 2026-03-08 06:00 to 20:00
```

---

## 📞 Support

**Need Help?**
- 📋 Check [config_templates/README.md](config_templates/README.md) for detailed configuration guide
- 🔧 Try `--validate` flag to test configurations
- 📊 Use `--verbose` flag for detailed logging

**Questions?** Contact the optimization team with:
- Your configuration file
- Error messages (if any)
- Description of your setup

---

**🎉 Transform your battery operations today!**  
*Optimal decisions, automated 24/7* ⚡