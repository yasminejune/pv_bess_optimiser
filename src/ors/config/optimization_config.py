"""Configuration schema for optimization runs.

Defines the structure and validation for JSON configuration files
that clients can fill out to run optimizations without code changes.
"""

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class PVConfiguration(BaseModel):
    """PV system configuration for optimization."""

    # Required PV specifications
    rated_power_kw: float = Field(..., description="Maximum rated power output in kW")
    max_export_kw: float | None = Field(
        None, description="Maximum power export to grid (kW). None = unlimited"
    )
    panel_area_m2: float | None = Field(
        None, description="Total panel surface area (m²). Required for weather-based forecasting"
    )
    panel_efficiency: float = Field(0.18, description="Panel efficiency ratio (0-1). Default 18%")

    # Generation data source
    generation_source: Literal["historical", "forecast", "manual"] = Field(
        ..., description="Source of generation data"
    )

    # Historical data path (if using historical)
    historical_data_path: str | None = Field(
        None, description="Path to historical PV generation CSV"
    )

    # Manual generation profile (if using manual)
    manual_generation_profile: dict[str, float] | None = Field(
        None, description="Hourly generation profile {HH:MM: power_kw}"
    )

    # Weather-based forecasting (if using forecast)
    location_lat: float | None = Field(None, description="Latitude for weather forecasting")
    location_lon: float | None = Field(None, description="Longitude for weather forecasting")

    # Optional constraints
    min_generation_kw: float = Field(0.0, description="Minimum generation threshold (kW)")
    curtailment_supported: bool = Field(True, description="Whether system supports curtailment")

    @field_validator("rated_power_kw", "max_export_kw")
    @classmethod
    def validate_positive_power(cls, v: float | None) -> float | None:
        """Validate that power values are positive."""
        if v is not None and v <= 0:
            raise ValueError("Power values must be positive")
        return v

    @field_validator("panel_efficiency")
    @classmethod
    def validate_efficiency(cls, v: float) -> float:
        """Validate that efficiency values are between 0 and 1."""
        if v <= 0 or v > 1:
            raise ValueError("Panel efficiency must be between 0 and 1")
        return v


class BatteryConfiguration(BaseModel):
    """Battery system configuration for optimization."""

    # Required battery specifications
    rated_power_mw: float = Field(..., description="Maximum rated power in/out (MW)")
    energy_capacity_mwh: float = Field(..., description="Total energy capacity (MWh)")

    # Operating constraints
    min_soc_percent: float = Field(10.0, description="Minimum state of charge (%)")
    max_soc_percent: float = Field(90.0, description="Maximum state of charge (%)")
    max_cycles_per_day: int = Field(3, description="Maximum charge/discharge cycles per day")

    # Efficiency parameters
    charge_efficiency: float = Field(0.97, description="Charging efficiency (0-1)")
    discharge_efficiency: float = Field(0.97, description="Discharging efficiency (0-1)")
    auxiliary_power_mw: float = Field(0.5, description="Auxiliary power consumption (MW)")
    self_discharge_rate_per_hour: float = Field(0.0005, description="Self-discharge rate per hour")

    # Current state
    current_energy_mwh: float | None = Field(
        None, description="Current stored energy (MWh). None = use 50%"
    )
    current_soc_percent: float | None = Field(
        None, description="Current SOC (%). Alternative to energy"
    )
    current_power_mw: float = Field(
        0.0, description="Current power flow (MW). +charging, -discharging"
    )
    current_mode: Literal["charging", "discharging", "idle"] = Field(
        "idle", description="Current operating mode"
    )
    cycles_used_today: int = Field(0, description="Charge/discharge cycles already used today")
    is_available: bool = Field(True, description="Whether battery is available for optimization")

    @field_validator("rated_power_mw", "energy_capacity_mwh")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        """Validate that values are positive."""
        if v <= 0:
            raise ValueError("Power and energy values must be positive")
        return v

    @field_validator("min_soc_percent", "max_soc_percent", "current_soc_percent")
    @classmethod
    def validate_soc_range(cls, v: float | None) -> float | None:
        """Validate that SOC percentages are between 0 and 100."""
        if v is not None and (v < 0 or v > 100):
            raise ValueError("SOC percentages must be between 0 and 100")
        return v

    @field_validator("charge_efficiency", "discharge_efficiency")
    @classmethod
    def validate_efficiency(cls, v: float) -> float:
        """Validate that efficiency values are between 0 and 1."""
        if v <= 0 or v > 1:
            raise ValueError("Efficiency values must be between 0 and 1")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Additional validation after model creation."""
        if self.min_soc_percent > self.max_soc_percent:
            raise ValueError("min_soc_percent must be <= max_soc_percent")

        if self.current_energy_mwh is not None:
            min_energy = self.energy_capacity_mwh * self.min_soc_percent / 100
            max_energy = self.energy_capacity_mwh * self.max_soc_percent / 100
            if not (min_energy <= self.current_energy_mwh <= max_energy):
                raise ValueError(
                    f"current_energy_mwh must be between {min_energy} and {max_energy} MWh"
                )


class OptimizationConfiguration(BaseModel):
    """Optimization period and parameters."""

    # Time period
    optimization_date: date = Field(..., description="Date for optimization (YYYY-MM-DD)")
    start_time: str = Field("00:00", description="Start time (HH:MM)")
    duration_hours: int = Field(24, description="Optimization duration in hours")
    time_step_minutes: int = Field(15, description="Time step resolution in minutes")

    # Price data source
    price_source: Literal["historical", "forecast", "manual"] = Field(
        "forecast", description="Source of price data"
    )
    historical_price_path: str | None = Field(None, description="Path to historical price CSV")
    manual_price_profile: dict[str, float] | None = Field(
        None, description="Hourly price profile {HH:MM: price_gbp_per_mwh}"
    )

    # Terminal pricing for battery valuation
    terminal_price_method: Literal["average", "manual"] = Field(
        "average", description="How to value terminal battery energy"
    )
    terminal_price_gbp_per_mwh: float | None = Field(
        None, description="Manual terminal price. Required if terminal_price_method='manual'"
    )

    @field_validator("duration_hours")
    @classmethod
    def validate_duration(cls, v: int) -> int:
        """Validate that duration is between 1 and 48 hours."""
        if v <= 0 or v > 48:
            raise ValueError("Duration must be between 1 and 48 hours")
        return v

    @field_validator("time_step_minutes")
    @classmethod
    def validate_time_step(cls, v: int) -> int:
        """Validate that time step is 15, 30, or 60 minutes."""
        if v not in [15, 30, 60]:
            raise ValueError("Time step must be 15, 30, or 60 minutes")
        return v

    @field_validator("start_time")
    @classmethod
    def validate_time_format(cls, v: str) -> str:
        """Validate that time format is HH:MM."""
        try:
            datetime.strptime(v, "%H:%M")
        except ValueError as e:
            raise ValueError("start_time must be in HH:MM format") from e
        return v


class OutputConfiguration(BaseModel):
    """Output file and display preferences."""

    # Output paths
    output_csv_path: str = Field("optimization_results.csv", description="Path for output CSV")
    detailed_log_path: str | None = Field(None, description="Path for detailed battery logs")

    # Display options
    include_summary: bool = Field(True, description="Include optimization summary")
    include_recommendations: bool = Field(True, description="Include action recommendations")
    verbose: bool = Field(False, description="Verbose logging during optimization")

    # Currency and units
    currency: str = Field("GBP", description="Currency for financial results")
    energy_units: str = Field("MWh", description="Energy units for display")
    power_units: str = Field("MW", description="Power units for display")


class OptimizationConfig(BaseModel):
    """Complete configuration for optimization run."""

    # Configuration metadata
    config_name: str = Field(..., description="Name/description of this configuration")
    created_by: str = Field("client", description="Who created this configuration")
    created_date: str | None = Field(
        None, description="When configuration was created (YYYY-MM-DD)"
    )

    # Core configurations
    pv: PVConfiguration | None = Field(None, description="PV system configuration. None = no PV")
    battery: BatteryConfiguration = Field(..., description="Battery system configuration")
    optimization: OptimizationConfiguration = Field(..., description="Optimization parameters")
    output: OutputConfiguration = Field(
        default=OutputConfiguration(), description="Output preferences"  # type: ignore[call-arg]
    )

    # Additional metadata
    notes: str = Field("", description="Additional notes or comments")

    def model_post_init(self, __context: Any) -> None:
        """Set default created_date if not provided."""
        if self.created_date is None:
            self.created_date = datetime.now().strftime("%Y-%m-%d")

    @property
    def has_pv(self) -> bool:
        """Check if PV system is configured."""
        return self.pv is not None

    @property
    def optimization_start_datetime(self) -> datetime:
        """Get the optimization start datetime."""
        return datetime.combine(
            self.optimization.optimization_date,
            datetime.strptime(self.optimization.start_time, "%H:%M").time(),
        )

    @property
    def optimization_end_datetime(self) -> datetime:
        """Get the optimization end datetime."""
        from datetime import timedelta

        return self.optimization_start_datetime + timedelta(hours=self.optimization.duration_hours)

    @property
    def total_time_steps(self) -> int:
        """Calculate total number of time steps."""
        return (self.optimization.duration_hours * 60) // self.optimization.time_step_minutes


# Utility functions for loading and validating configurations
def load_config_from_json(json_path: str) -> OptimizationConfig:
    """Load and validate configuration from JSON file.

    Args:
        json_path: Path to JSON configuration file

    Returns:
        Validated OptimizationConfig object

    Raises:
        FileNotFoundError: If JSON file doesn't exist
        ValidationError: If configuration is invalid
    """
    import json
    from pathlib import Path

    config_file = Path(json_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {json_path}")

    with open(config_file, encoding="utf-8") as f:
        config_data = json.load(f)

    return OptimizationConfig.model_validate(config_data)


def save_config_to_json(config: OptimizationConfig, json_path: str) -> None:
    """Save configuration to JSON file.

    Args:
        config: OptimizationConfig object to save
        json_path: Output path for JSON file
    """
    import json
    from pathlib import Path

    # Ensure directory exists
    output_path = Path(json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict and save with pretty formatting
    config_dict = config.model_dump(exclude_none=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)
