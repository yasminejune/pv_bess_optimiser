"""Service for updating PV system state from telemetry."""

from ors.domain.models.pv import PVSpec, PVState, PVTelemetry


def estimate_energy_from_radiance(
    solar_radiance_kw_per_m2: float,
    panel_area_m2: float,
    panel_efficiency: float,
    timestep_minutes: int = 15,
) -> float:
    """Estimate PV energy production from solar radiance.

    Formula: energy_kwh = solar_radiance * panel_area * efficiency * time_interval_hours

    Args:
        solar_radiance_kw_per_m2: Solar radiance (kW/m²)
        panel_area_m2: Total panel surface area (m²)
        panel_efficiency: Panel efficiency ratio (0-1)
        timestep_minutes: Time step duration in minutes (default: 15)

    Returns:
        Estimated energy production (kWh)

    Raises:
        ValueError: If any parameter is out of valid range

    """
    if timestep_minutes <= 0:
        raise ValueError(f"timestep_minutes must be positive, got {timestep_minutes}")
    if solar_radiance_kw_per_m2 < 0:
        raise ValueError(
            f"solar_radiance_kw_per_m2 must be non-negative, got {solar_radiance_kw_per_m2}"
        )
    if panel_area_m2 <= 0:
        raise ValueError(f"panel_area_m2 must be positive, got {panel_area_m2}")
    if not (0 <= panel_efficiency <= 1):
        raise ValueError(f"panel_efficiency must be between 0 and 1, got {panel_efficiency}")

    time_interval_hours = timestep_minutes / 60.0
    energy_kwh = solar_radiance_kw_per_m2 * panel_area_m2 * panel_efficiency * time_interval_hours

    return energy_kwh


def update_pv_state(
    spec: PVSpec,
    telemetry: PVTelemetry,
    timestep_minutes: int = 15,
    *,
    prev_state: PVState | None = None,
) -> PVState:
    """Update PV state from telemetry and specification.

    Args:
        spec: PV system specification
        telemetry: Raw telemetry data
        timestep_minutes: Time step duration in minutes (default: 15)
        prev_state: Optional previous state (unused, reserved for future)

    Returns:
        Validated and processed PV state

    Raises:
        ValueError: If timestep_minutes <= 0

    """
    if timestep_minutes <= 0:
        raise ValueError(f"timestep_minutes must be positive, got {timestep_minutes}")

    quality_flags: set[str] = set()
    estimated_from_radiance = False

    # Determine generation_kw and energy_kwh
    if telemetry.generation_kw is not None:
        # Primary source: telemetry
        generation_kw = telemetry.generation_kw
        energy_kwh = generation_kw * (timestep_minutes / 60.0)
        estimated_from_radiance = False
    else:
        # Missing generation
        quality_flags.add("missing_generation")

        # Try radiance-based estimation
        if (
            telemetry.solar_radiance_kw_per_m2 is not None
            and spec.panel_area_m2 is not None
            and spec.panel_efficiency is not None
        ):

            energy_kwh = estimate_energy_from_radiance(
                telemetry.solar_radiance_kw_per_m2,
                spec.panel_area_m2,
                spec.panel_efficiency,
                timestep_minutes,
            )
            generation_kw = energy_kwh / (timestep_minutes / 60.0)
            estimated_from_radiance = True
            quality_flags.add("estimated_from_radiance")
        else:
            # No radiance info available
            generation_kw = 0.0
            energy_kwh = 0.0
            estimated_from_radiance = False

    # Validate and clamp generation_kw
    if generation_kw < 0:
        generation_kw = 0.0
        quality_flags.add("negative_generation_clamped")

    # Apply rated power constraint
    if generation_kw > spec.rated_power_kw:
        generation_kw = spec.rated_power_kw
        quality_flags.add("above_rated_clamped")

    # Apply minimum generation constraint
    if generation_kw < spec.min_generation_kw:
        generation_kw = spec.min_generation_kw
        quality_flags.add("below_min_generation_clamped")

    # Recompute energy_kwh after clamping
    energy_kwh = generation_kw * (timestep_minutes / 60.0)

    # Compute exportable power and energy
    if spec.max_export_kw is None:
        exportable_kw = generation_kw
    else:
        exportable_kw = min(generation_kw, spec.max_export_kw)

    exportable_kwh = exportable_kw * (timestep_minutes / 60.0)

    # Compute curtailment
    curtailed_kw = 0.0
    curtailed = False

    if spec.max_export_kw is not None and generation_kw > spec.max_export_kw:
        if spec.curtailment_supported:
            curtailed_kw = generation_kw - spec.max_export_kw
            curtailed = True
        else:
            quality_flags.add("export_cap_applied_no_curtailment")

    return PVState(
        timestamp=telemetry.timestamp,
        generation_kw=generation_kw,
        energy_kwh=energy_kwh,
        curtailed_kw=curtailed_kw,
        curtailed=curtailed,
        exportable_kw=exportable_kw,
        exportable_kwh=exportable_kwh,
        estimated_from_radiance=estimated_from_radiance,
        quality_flags=quality_flags,
    )
