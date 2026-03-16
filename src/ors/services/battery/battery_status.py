"""Service for updating battery system state from telemetry."""

from ors.domain.models.battery import BatterySpec, BatteryState, BatteryTelemetry


def estimate_energy_from_soc(
    soc_percent: float,
    energy_capacity_mwh: float,
) -> float:
    """Estimate battery energy from state of charge percentage.

    Args:
        soc_percent (float): State of charge percentage (0-100)
        energy_capacity_mwh (float): Battery energy capacity (MWh)

    Returns:
        float: Estimated stored energy (MWh)

    Raises:
        ValueError: If parameters are out of valid range
    """
    if not (0 <= soc_percent <= 100):
        raise ValueError(f"soc_percent must be between 0 and 100, got {soc_percent}")
    if energy_capacity_mwh <= 0:
        raise ValueError(f"energy_capacity_mwh must be positive, got {energy_capacity_mwh}")

    return energy_capacity_mwh * soc_percent / 100.0


def estimate_soc_from_energy(
    energy_mwh: float,
    energy_capacity_mwh: float,
) -> float:
    """Estimate state of charge percentage from stored energy.

    Args:
        energy_mwh (float): Current stored energy (MWh)
        energy_capacity_mwh (float): Battery energy capacity (MWh)

    Returns:
        float: Estimated SOC percentage (0-100)

    Raises:
        ValueError: If parameters are out of valid range
    """
    if energy_mwh < 0:
        raise ValueError(f"energy_mwh must be non-negative, got {energy_mwh}")
    if energy_capacity_mwh <= 0:
        raise ValueError(f"energy_capacity_mwh must be positive, got {energy_capacity_mwh}")

    return (energy_mwh / energy_capacity_mwh) * 100.0


def determine_operating_mode(power_mw: float, power_threshold_mw: float = 0.1) -> str:
    """Determine battery operating mode from power flow.

    Args:
        power_mw (float): Power flow (MW). Positive = charging, Negative = discharging
        power_threshold_mw (float): Minimum power threshold for non-idle mode (default: 0.1 MW)

    Returns:
        str: Operating mode: "charging", "discharging", or "idle"
    """
    if abs(power_mw) < power_threshold_mw:
        return "idle"
    elif power_mw > 0:
        return "charging"
    else:
        return "discharging"


def update_battery_state(
    spec: BatterySpec,
    telemetry: BatteryTelemetry,
    *,
    prev_state: BatteryState | None = None,
    power_threshold_mw: float = 0.1,
) -> BatteryState:
    """Update battery state from telemetry and specification.

    Processes raw telemetry data to create validated battery state with
    quality flags for missing data, estimations, or constraint violations.

    Args:
        spec (BatterySpec): Battery system specification
        telemetry (BatteryTelemetry): Raw battery telemetry data
        prev_state (BatteryState | None): Optional previous state for consistency checks
        power_threshold_mw (float): Power threshold for determining idle mode (default: 0.1 MW)

    Returns:
        BatteryState: Validated and processed battery state

    Raises:
        ValueError: If both energy and SOC are missing and no previous state available
    """
    quality_flags: set[str] = set()
    estimated_values: set[str] = set()

    # Determine energy and SOC (prioritize energy over SOC for accuracy)
    energy_mwh = None
    soc_percent = None

    if telemetry.current_energy_mwh is not None:
        # Primary source: direct energy measurement
        energy_mwh = telemetry.current_energy_mwh
        soc_percent = estimate_soc_from_energy(energy_mwh, spec.energy_capacity_mwh)

        # Check if telemetry also provides SOC and if they're consistent
        if telemetry.current_soc_percent is not None:
            expected_soc = estimate_soc_from_energy(energy_mwh, spec.energy_capacity_mwh)
            soc_diff = abs(telemetry.current_soc_percent - expected_soc)
            if soc_diff > 1.0:  # More than 1% difference
                quality_flags.add("energy_soc_mismatch")

    elif telemetry.current_soc_percent is not None:
        # Secondary source: SOC percentage
        soc_percent = telemetry.current_soc_percent
        energy_mwh = estimate_energy_from_soc(soc_percent, spec.energy_capacity_mwh)
        estimated_values.add("energy_mwh")
        quality_flags.add("energy_estimated_from_soc")

    else:
        # Missing both energy and SOC
        quality_flags.add("missing_energy_and_soc")

        if prev_state is not None:
            # Use previous state as fallback
            energy_mwh = prev_state.energy_mwh
            soc_percent = prev_state.soc_percent
            estimated_values.update(["energy_mwh", "soc_percent"])
            quality_flags.add("using_previous_state")
        else:
            # No previous state available - use safe default (50% SOC)
            soc_percent = 50.0
            energy_mwh = estimate_energy_from_soc(soc_percent, spec.energy_capacity_mwh)
            estimated_values.update(["energy_mwh", "soc_percent"])
            quality_flags.add("defaulted_to_50_percent_soc")

    # Validate and clamp energy within bounds
    min_energy = spec.min_energy_mwh
    max_energy = spec.max_energy_mwh

    if energy_mwh < min_energy:
        energy_mwh = min_energy
        soc_percent = estimate_soc_from_energy(energy_mwh, spec.energy_capacity_mwh)
        quality_flags.add("energy_clamped_to_minimum")
    elif energy_mwh > max_energy:
        energy_mwh = max_energy
        soc_percent = estimate_soc_from_energy(energy_mwh, spec.energy_capacity_mwh)
        quality_flags.add("energy_clamped_to_maximum")

    # Determine power flow
    power_mw = 0.0
    if telemetry.current_power_mw is not None:
        power_mw = telemetry.current_power_mw

        # Validate power limits
        if abs(power_mw) > spec.rated_power_mw:
            power_mw = spec.rated_power_mw if power_mw > 0 else -spec.rated_power_mw
            quality_flags.add("power_clamped_to_rated")
    else:
        quality_flags.add("missing_power_data")
        estimated_values.add("power_mw")

    # Determine operating mode
    operating_mode = "idle"
    if telemetry.operating_mode is not None:
        operating_mode = telemetry.operating_mode

        # Cross-validate mode with power measurement
        if telemetry.current_power_mw is not None:
            power_based_mode = determine_operating_mode(power_mw, power_threshold_mw)
            if operating_mode != power_based_mode:
                quality_flags.add("mode_power_mismatch")
                # Trust power measurement over reported mode
                operating_mode = power_based_mode
    else:
        operating_mode = determine_operating_mode(power_mw, power_threshold_mw)
        estimated_values.add("operating_mode")
        quality_flags.add("mode_estimated_from_power")

    # Check availability
    is_available = telemetry.is_available

    # Additional validation checks
    if prev_state is not None:
        # Check for unrealistic state changes
        energy_change = abs(energy_mwh - prev_state.energy_mwh)
        time_diff = (telemetry.timestamp - prev_state.timestamp).total_seconds() / 3600.0  # hours

        if time_diff > 0:
            # Maximum possible energy change based on rated power
            max_possible_change = spec.rated_power_mw * time_diff
            if energy_change > max_possible_change * 1.1:  # 10% tolerance
                quality_flags.add("unrealistic_energy_change")

    # Ensure operating_mode is valid Literal type
    valid_modes = {"charging", "discharging", "idle"}
    if operating_mode not in valid_modes:
        operating_mode = "idle"  # Default fallback
        quality_flags.add("invalid_mode_defaulted_to_idle")

    return BatteryState(
        timestamp=telemetry.timestamp,
        energy_mwh=energy_mwh,
        soc_percent=soc_percent,
        power_mw=power_mw,
        operating_mode=operating_mode,  # type: ignore[arg-type]
        is_available=is_available,
        estimated_values=estimated_values,
        quality_flags=quality_flags,
    )
