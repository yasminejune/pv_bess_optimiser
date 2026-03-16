"""Domain models for battery energy storage system.

This module contains domain models for battery system specifications,
real-time telemetry, and state representation following the same pattern
as the PV domain models.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, field_validator


class BatterySpec(BaseModel):
    """Battery system specification.

    Defines the physical characteristics and operational constraints
    of a battery energy storage system.
    """

    rated_power_mw: float
    """Maximum rated power input/output in MW"""

    energy_capacity_mwh: float
    """Total energy capacity in MWh"""

    min_soc_percent: float = 10.0
    """Minimum allowable state of charge percentage (0-100)"""

    max_soc_percent: float = 90.0
    """Maximum allowable state of charge percentage (0-100)"""

    charge_efficiency: float = 0.97
    """Charging efficiency (0-1)"""

    discharge_efficiency: float = 0.97
    """Discharging efficiency (0-1)"""

    auxiliary_power_mw: float = 0.5
    """Auxiliary power consumption in MW"""

    self_discharge_rate_per_hour: float = 0.0005
    """Self-discharge rate per hour (0-1)"""

    @field_validator("rated_power_mw", "energy_capacity_mwh")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        """Validate that power and energy values are positive."""
        if v <= 0:
            raise ValueError("Power and energy values must be positive")
        return v

    @field_validator("min_soc_percent", "max_soc_percent")
    @classmethod
    def validate_soc_range(cls, v: float) -> float:
        """Validate that SOC percentages are between 0 and 100."""
        if v < 0 or v > 100:
            raise ValueError("SOC percentages must be between 0 and 100")
        return v

    @field_validator("charge_efficiency", "discharge_efficiency")
    @classmethod
    def validate_efficiency(cls, v: float) -> float:
        """Validate that efficiency values are between 0 and 1."""
        if v <= 0 or v > 1:
            raise ValueError("Efficiency values must be between 0 and 1")
        return v

    @field_validator("auxiliary_power_mw", "self_discharge_rate_per_hour")
    @classmethod
    def validate_non_negative(cls, v: float) -> float:
        """Validate that auxiliary power and self-discharge rate are non-negative."""
        if v < 0:
            raise ValueError("Auxiliary power and self-discharge rate must be non-negative")
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate that min_soc <= max_soc."""
        if self.min_soc_percent > self.max_soc_percent:
            raise ValueError("min_soc_percent must be <= max_soc_percent")

    @property
    def min_energy_mwh(self) -> float:
        """Minimum energy in MWh."""
        return self.energy_capacity_mwh * self.min_soc_percent / 100.0

    @property
    def max_energy_mwh(self) -> float:
        """Maximum energy in MWh."""
        return self.energy_capacity_mwh * self.max_soc_percent / 100.0


class BatteryTelemetry(BaseModel):
    """Battery telemetry data from real-time sensors or monitoring systems.

    Represents actual current state of the battery as measured or reported
    by the battery management system.
    """

    timestamp: datetime
    """Measurement or report timestamp"""

    current_energy_mwh: float | None = None
    """Current stored energy in MWh. None indicates missing telemetry."""

    current_soc_percent: float | None = None
    """Current state of charge percentage (0-100). None indicates missing telemetry."""

    current_power_mw: float | None = None
    """Current power flow in MW. Positive = charging, Negative = discharging, 0 = idle."""

    operating_mode: Literal["charging", "discharging", "idle"] | None = None
    """Current operating mode. None indicates unknown or missing telemetry."""

    is_available: bool = True
    """Whether battery is available for optimization control"""

    @field_validator("current_soc_percent")
    @classmethod
    def validate_soc(cls, v: float | None) -> float | None:
        """Validate that SOC percentage is between 0 and 100."""
        if v is not None and (v < 0 or v > 100):
            raise ValueError("SOC percentage must be between 0 and 100")
        return v

    def get_energy_from_soc(self, spec: BatterySpec) -> float | None:
        """Calculate energy from SOC percentage using battery spec.

        Args:
            spec: Battery specification containing capacity

        Returns:
            Energy in MWh, or None if SOC is not available
        """
        if self.current_soc_percent is None:
            return None
        return spec.energy_capacity_mwh * self.current_soc_percent / 100.0

    def get_soc_from_energy(self, spec: BatterySpec) -> float | None:
        """Calculate SOC percentage from energy using battery spec.

        Args:
            spec: Battery specification containing capacity

        Returns:
            SOC percentage (0-100), or None if energy is not available
        """
        if self.current_energy_mwh is None:
            return None
        return (self.current_energy_mwh / spec.energy_capacity_mwh) * 100.0


class BatteryState(BaseModel):
    """Validated and processed battery state with derived values and quality indicators.

    Represents the cleaned, validated battery state after processing
    telemetry data with quality flags for any issues or estimations.
    """

    timestamp: datetime
    """State timestamp"""

    energy_mwh: float
    """Final validated stored energy (MWh)"""

    soc_percent: float
    """Final validated state of charge percentage (0-100)"""

    power_mw: float
    """Current power flow (MW). Positive = charging, Negative = discharging, 0 = idle"""

    operating_mode: Literal["charging", "discharging", "idle"]
    """Current operating mode"""

    is_available: bool
    """Whether battery is available for optimization"""

    estimated_values: set[str]
    """Set of field names that were estimated rather than measured"""

    quality_flags: set[str]
    """Quality flags indicating any issues or processing notes"""

    @field_validator("soc_percent")
    @classmethod
    def validate_soc(cls, v: float) -> float:
        """Validate that SOC percentage is between 0 and 100."""
        if v < 0 or v > 100:
            raise ValueError("SOC percentage must be between 0 and 100")
        return v

    @property
    def has_quality_issues(self) -> bool:
        """Whether this state has any quality issues."""
        return len(self.quality_flags) > 0

    @property
    def has_estimated_values(self) -> bool:
        """Whether this state contains estimated values."""
        return len(self.estimated_values) > 0
