"""PV system domain models for the Operational Recommendation System."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PVSpec:
    """PV system specification and constraints.

    Attributes:
        rated_power_kw: Maximum rated power output (kW)
        max_export_kw: Maximum power that can be exported (kW). None means unlimited.
        min_generation_kw: Minimum generation threshold (kW)
        curtailment_supported: Whether the system can curtail excess generation
        panel_area_m2: Total panel surface area (m²). Optional, used for radiance estimation.
        panel_efficiency: Panel efficiency ratio (0-1). Optional, used for radiance estimation.
        dc_capacity_kw: DC capacity of PV array (kW). Optional.
        ac_capacity_kw: AC capacity of inverter (kW). Optional.
        dc_ac_ratio: DC to AC capacity ratio. Optional.
        inverter_efficiency: Inverter efficiency (0-1). Optional.
        performance_ratio: Overall system performance ratio (0-1). Optional.
        degradation_per_year: Annual degradation rate (%). Optional.
        clipping_loss_factor: Clipping loss factor (0-1). Optional.
        availability: System availability factor (0-1). Optional.
        forced_outage_duration_h: Expected forced outage duration (hours). Optional.

    """

    rated_power_kw: float
    max_export_kw: float | None
    min_generation_kw: float = 0.0
    curtailment_supported: bool = True
    panel_area_m2: float | None = None
    panel_efficiency: float | None = None
    dc_capacity_kw: float | None = None
    ac_capacity_kw: float | None = None
    dc_ac_ratio: float | None = None
    inverter_efficiency: float | None = None
    performance_ratio: float | None = None
    degradation_per_year: float | None = None
    clipping_loss_factor: float | None = None
    availability: float | None = None
    forced_outage_duration_h: float | None = None

    def __post_init__(self) -> None:
        """Validate PV specification parameters."""
        if self.rated_power_kw <= 0:
            raise ValueError("rated_power_kw must be positive")
        if self.max_export_kw is not None and self.max_export_kw < 0:
            raise ValueError("max_export_kw must be non-negative")
        if self.min_generation_kw < 0:
            raise ValueError("min_generation_kw must be non-negative")
        if self.panel_area_m2 is not None and self.panel_area_m2 <= 0:
            raise ValueError("panel_area_m2 must be positive")
        if self.panel_efficiency is not None and not (0 <= self.panel_efficiency <= 1):
            raise ValueError("panel_efficiency must be between 0 and 1")
        if self.dc_capacity_kw is not None and self.dc_capacity_kw <= 0:
            raise ValueError("dc_capacity_kw must be positive")
        if self.ac_capacity_kw is not None and self.ac_capacity_kw <= 0:
            raise ValueError("ac_capacity_kw must be positive")
        if self.dc_ac_ratio is not None and self.dc_ac_ratio <= 0:
            raise ValueError("dc_ac_ratio must be positive")
        if self.inverter_efficiency is not None and not (0 <= self.inverter_efficiency <= 1):
            raise ValueError("inverter_efficiency must be between 0 and 1")
        if self.performance_ratio is not None and not (0 <= self.performance_ratio <= 1):
            raise ValueError("performance_ratio must be between 0 and 1")
        if self.degradation_per_year is not None and self.degradation_per_year < 0:
            raise ValueError("degradation_per_year must be non-negative")
        if self.clipping_loss_factor is not None and not (0 <= self.clipping_loss_factor <= 1):
            raise ValueError("clipping_loss_factor must be between 0 and 1")
        if self.availability is not None and not (0 <= self.availability <= 1):
            raise ValueError("availability must be between 0 and 1")
        if self.forced_outage_duration_h is not None and self.forced_outage_duration_h < 0:
            raise ValueError("forced_outage_duration_h must be non-negative")


@dataclass
class PVTelemetry:
    """PV telemetry data from sensors or forecasts.

    Attributes:
        timestamp: Measurement timestamp
        generation_kw: Instantaneous measured PV output (kW). None if missing.
        solar_radiance_kw_per_m2: Solar radiance (kW/m²). Optional, used for estimation when generation is missing.

    """

    timestamp: datetime
    generation_kw: float | None
    solar_radiance_kw_per_m2: float | None = None


@dataclass
class PVState:
    """Current PV system state with derived values.

    Attributes:
        timestamp: State timestamp
        generation_kw: Final cleaned/derived generation power (kW)
        energy_kwh: Energy produced this timestep (kWh)
        curtailed_kw: Amount of power curtailed (kW)
        curtailed: Whether curtailment is active
        exportable_kw: Power available for export (kW)
        exportable_kwh: Energy available for export this timestep (kWh)
        estimated_from_radiance: Whether generation was estimated from radiance
        quality_flags: Set of quality issue flags

    """

    timestamp: datetime
    generation_kw: float
    energy_kwh: float
    curtailed_kw: float
    curtailed: bool
    exportable_kw: float
    exportable_kwh: float
    estimated_from_radiance: bool
    quality_flags: set[str] = field(default_factory=set)
