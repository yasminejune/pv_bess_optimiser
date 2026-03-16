"""Battery management utilities and simulation support.

This module contains:
- Core battery physics: BatteryParams, step_energy, compute_losses
- Configuration management: load battery parameters from JSON
- CSV logging utilities: export simulation data

All calculations are consistent with the project energy equation:

    E_t = E_{t-1}
          + eta_ch * P_ch * dt
          - (P_dis * dt) / eta_dis
          - P_aux * dt
          - E_{t-1} * r_sd * dt

Where:
- P_ch = P_grid + P_sol
- dt is in hours
- power is in MW
- energy is in MWh


Usage:
    from battery_management import (
        BatteryParams, step_energy, compute_losses,
        load_battery_params, write_simulation_csv
    )
"""

import csv
import json
from pathlib import Path
from typing import Any


class BatteryParams:
    """Battery parameters.

    Notes:
        - Power is in MW
        - Energy is in MWh
        - dt_hours is in hours
        - r_sd_per_hour is per hour

    This is implemented without `@dataclass`. Instances are immutable after init.
    All inputs are validated for physical consistency:
        - p_rated_mw must be > 0
        - eta_ch, eta_dis must be in (0, 1] (efficiencies between 0% and 100%)
        - a_aux, r_sd_per_hour must be >= 0
        - e_duration_hours must be > 0
        - e_min_frac, e_max_frac must be in [0, 1]
        - e_min_frac must be <= e_max_frac

    Raises:
        ValueError: If any parameter fails validation
    """

    __slots__ = (
        "p_rated_mw",
        "eta_ch",
        "eta_dis",
        "a_aux",
        "r_sd_per_hour",
        "e_duration_hours",
        "e_min_frac",
        "e_max_frac",
        "_frozen",
    )

    # Type annotations for __slots__ attributes
    p_rated_mw: float
    eta_ch: float
    eta_dis: float
    a_aux: float
    r_sd_per_hour: float
    e_duration_hours: float
    e_min_frac: float
    e_max_frac: float
    _frozen: bool

    def __init__(
        self,
        p_rated_mw: float = 100.0,
        eta_ch: float = 0.97,
        eta_dis: float = 0.97,
        a_aux: float = 0.005,
        r_sd_per_hour: float = 0.0005,
        e_duration_hours: float = 3.0,
        e_min_frac: float = 0.10,
        e_max_frac: float = 0.90,
    ) -> None:
        # Validate inputs for physical consistency
        if p_rated_mw <= 0:
            raise ValueError("p_rated_mw must be > 0")
        if eta_ch <= 0 or eta_ch > 1:
            raise ValueError("eta_ch must be in (0, 1]")
        if eta_dis <= 0 or eta_dis > 1:
            raise ValueError("eta_dis must be in (0, 1]")
        if a_aux < 0:
            raise ValueError("a_aux must be >= 0")
        if r_sd_per_hour < 0:
            raise ValueError("r_sd_per_hour must be >= 0")
        if e_duration_hours <= 0:
            raise ValueError("e_duration_hours must be > 0")
        if e_min_frac < 0 or e_min_frac > 1:
            raise ValueError("e_min_frac must be in [0, 1]")
        if e_max_frac < 0 or e_max_frac > 1:
            raise ValueError("e_max_frac must be in [0, 1]")
        if e_min_frac > e_max_frac:
            raise ValueError("e_min_frac must be <= e_max_frac")

        object.__setattr__(self, "p_rated_mw", float(p_rated_mw))
        object.__setattr__(self, "eta_ch", float(eta_ch))
        object.__setattr__(self, "eta_dis", float(eta_dis))
        object.__setattr__(self, "a_aux", float(a_aux))
        object.__setattr__(self, "r_sd_per_hour", float(r_sd_per_hour))
        object.__setattr__(self, "e_duration_hours", float(e_duration_hours))
        object.__setattr__(self, "e_min_frac", float(e_min_frac))
        object.__setattr__(self, "e_max_frac", float(e_max_frac))
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("BatteryParams is immutable")
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return (
            "BatteryParams("
            f"p_rated_mw={self.p_rated_mw}, "
            f"eta_ch={self.eta_ch}, "
            f"eta_dis={self.eta_dis}, "
            f"a_aux={self.a_aux}, "
            f"r_sd_per_hour={self.r_sd_per_hour}, "
            f"e_duration_hours={self.e_duration_hours}, "
            f"e_min_frac={self.e_min_frac}, "
            f"e_max_frac={self.e_max_frac}"
            ")"
        )

    @property
    def p_aux_mw(self) -> float:
        """Constant auxiliary power draw (MW)."""
        return self.a_aux * self.p_rated_mw

    @property
    def e_cap_mwh(self) -> float:
        """Energy capacity (MWh)."""
        return self.p_rated_mw * self.e_duration_hours

    @property
    def e_min_mwh(self) -> float:
        """Minimum energy threshold in MWh."""
        return self.e_min_frac * self.e_cap_mwh

    @property
    def e_max_mwh(self) -> float:
        """Maximum energy threshold in MWh."""
        return self.e_max_frac * self.e_cap_mwh


class LossBreakdown:
    """Per-step energy loss components (MWh).

    Implemented without `@dataclass`. Instances are immutable after init.
    """

    __slots__ = (
        "loss_charge_ineff_mwh",
        "loss_discharge_ineff_mwh",
        "loss_aux_mwh",
        "loss_self_discharge_mwh",
        "_frozen",
    )

    # Type annotations for __slots__ attributes
    loss_charge_ineff_mwh: float
    loss_discharge_ineff_mwh: float
    loss_aux_mwh: float
    loss_self_discharge_mwh: float
    _frozen: bool

    def __init__(
        self,
        *,
        loss_charge_ineff_mwh: float,
        loss_discharge_ineff_mwh: float,
        loss_aux_mwh: float,
        loss_self_discharge_mwh: float,
    ) -> None:
        object.__setattr__(self, "loss_charge_ineff_mwh", loss_charge_ineff_mwh)
        object.__setattr__(self, "loss_discharge_ineff_mwh", loss_discharge_ineff_mwh)
        object.__setattr__(self, "loss_aux_mwh", loss_aux_mwh)
        object.__setattr__(self, "loss_self_discharge_mwh", loss_self_discharge_mwh)
        object.__setattr__(self, "_frozen", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("LossBreakdown is immutable")
        object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return (
            "LossBreakdown("
            f"loss_charge_ineff_mwh={self.loss_charge_ineff_mwh}, "
            f"loss_discharge_ineff_mwh={self.loss_discharge_ineff_mwh}, "
            f"loss_aux_mwh={self.loss_aux_mwh}, "
            f"loss_self_discharge_mwh={self.loss_self_discharge_mwh}"
            ")"
        )

    @property
    def total_loss_mwh(self) -> float:
        """Total energy losses across all categories in MWh."""
        return (
            self.loss_charge_ineff_mwh
            + self.loss_discharge_ineff_mwh
            + self.loss_aux_mwh
            + self.loss_self_discharge_mwh
        )


def clamp(value: float, lo: float, hi: float) -> float:
    """Clamp a scalar to [lo, hi].

    Args:
        value (float): Value to clamp
        lo (float): Lower bound
        hi (float): Upper bound

    Returns:
        float: Clamped value within bounds
    """
    return max(lo, min(hi, value))


def compute_losses(
    *,
    e_prev_mwh: float,
    p_grid_mw: float,
    p_sol_mw: float,
    p_dis_mw: float,
    params: BatteryParams,
    dt_hours: float,
) -> LossBreakdown:
    """Compute loss components for a single time step.

    Important:
        These losses are derived from the same update equation used in
        `step_energy`.

    Args:
        e_prev_mwh (float): Previous state-of-energy (MWh)
        p_grid_mw (float): Grid charging power (MW)
        p_sol_mw (float): Solar charging power (MW)
        p_dis_mw (float): Discharging power (MW)
        params (BatteryParams): Battery parameters
        dt_hours (float): Step duration in hours

    Returns:
        LossBreakdown: Loss components in MWh

    Raises:
        ValueError: If dt_hours <= 0
    """
    if dt_hours <= 0:
        raise ValueError("dt_hours must be > 0")

    p_ch_mw = max(0.0, p_grid_mw) + max(0.0, p_sol_mw)
    p_dis = max(0.0, p_dis_mw)

    # Charging inefficiency: ideal stored energy would be P_ch*dt, actual is eta_ch*P_ch*dt
    loss_charge = (1.0 - params.eta_ch) * p_ch_mw * dt_hours

    # Discharging inefficiency: battery must supply (P_dis*dt)/eta_dis to deliver P_dis*dt
    loss_discharge = (p_dis * dt_hours) * (1.0 / params.eta_dis - 1.0)

    # Auxiliary and self-discharge
    loss_aux = params.p_aux_mw * dt_hours
    loss_sd = max(0.0, e_prev_mwh) * params.r_sd_per_hour * dt_hours

    return LossBreakdown(
        loss_charge_ineff_mwh=float(loss_charge),
        loss_discharge_ineff_mwh=float(loss_discharge),
        loss_aux_mwh=float(loss_aux),
        loss_self_discharge_mwh=float(loss_sd),
    )


def step_energy(
    *,
    e_prev_mwh: float,
    p_grid_mw: float,
    p_sol_mw: float,
    p_dis_mw: float,
    params: BatteryParams,
    dt_hours: float,
    enforce_bounds: bool = True,
) -> float:
    """Update battery energy by one time step.

    Args:
        e_prev_mwh (float): Previous state-of-energy (MWh)
        p_grid_mw (float): Grid charging power (MW)
        p_sol_mw (float): Solar charging power (MW)
        p_dis_mw (float): Discharging power (MW)
        params (BatteryParams): Battery parameters
        dt_hours (float): Step duration in hours
        enforce_bounds (bool): If True, clamp E to [E_min, E_max]

    Returns:
        float: New energy state E_t in MWh

    Raises:
        ValueError: If dt_hours <= 0
    """
    if dt_hours <= 0:
        raise ValueError("dt_hours must be > 0")

    e_prev = float(e_prev_mwh)
    p_ch = max(0.0, p_grid_mw) + max(0.0, p_sol_mw)
    p_dis = max(0.0, p_dis_mw)

    e_next = (
        e_prev
        + params.eta_ch * p_ch * dt_hours
        - (p_dis * dt_hours) / params.eta_dis
        - params.p_aux_mw * dt_hours
        - e_prev * params.r_sd_per_hour * dt_hours
    )

    if enforce_bounds:
        e_next = clamp(e_next, params.e_min_mwh, params.e_max_mwh)

    return float(e_next)


# =============================================================================
# CONFIGURATION MANAGEMENT
# =============================================================================


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        config_path (str | Path): Path to JSON configuration file

    Returns:
        dict[str, Any]: Dictionary containing configuration data

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If JSON is invalid or missing required fields
    """
    try:
        with open(config_path) as f:
            config: dict[str, Any] = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from None
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in config file: {e}") from e

    # Validate required sections
    if "battery_params" not in config:
        raise ValueError("Missing 'battery_params' section in config")
    if "simulation_defaults" not in config:
        raise ValueError("Missing 'simulation_defaults' section in config")

    return dict[str, Any](config)


def load_battery_params(config_path: str | Path) -> BatteryParams:
    """Load BatteryParams from JSON config file.

    Args:
        config_path (str | Path): Path to JSON configuration file

    Returns:
        BatteryParams: BatteryParams object with loaded parameters

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid or missing required battery parameters
    """
    config = load_config(config_path)
    battery_config = config["battery_params"]

    # Validate all required parameters are present
    required_params = {
        "p_rated_mw",
        "eta_ch",
        "eta_dis",
        "a_aux",
        "r_sd_per_hour",
        "e_duration_hours",
        "e_min_frac",
        "e_max_frac",
    }
    missing = required_params - set(battery_config.keys())
    if missing:
        raise ValueError(f"Missing battery parameters in config: {missing}")

    return BatteryParams(**battery_config)


def load_simulation_defaults(config_path: str | Path) -> dict[str, Any]:
    """Load simulation defaults from JSON config file.

    Args:
        config_path (str | Path): Path to JSON configuration file

    Returns:
        dict[str, Any]: Dictionary containing simulation default values

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid or missing simulation defaults
    """
    config = load_config(config_path)
    sim_config: dict[str, Any] = config["simulation_defaults"]

    # Validate required simulation parameters
    required_sim_params = {"dt_hours", "enforce_bounds"}
    missing = required_sim_params - set(sim_config.keys())
    if missing:
        raise ValueError(f"Missing simulation parameters in config: {missing}")

    return dict[str, Any](sim_config)


def load_battery_params_and_defaults(
    config_path: str | Path,
) -> tuple[BatteryParams, dict[str, Any]]:
    """Load both battery parameters and simulation defaults from config file.

    Args:
        config_path (str | Path): Path to JSON configuration file

    Returns:
        tuple[BatteryParams, dict[str, Any]]: Tuple of (BatteryParams, simulation_defaults)
    """
    params = load_battery_params(config_path)
    defaults = load_simulation_defaults(config_path)
    return params, defaults


# =============================================================================
# CSV LOGGING UTILITIES
# =============================================================================


def write_simulation_csv(
    logs: list[dict[str, Any]], csv_path: str | Path = "battery_storage.csv"
) -> None:
    """Write simulation logs to CSV file.

    Args:
        logs (list[dict[str, Any]]): List of log dictionaries containing simulation data
        csv_path (str | Path): Path where CSV file should be written

    Expected log entry format:
        {
            'simulation_step': int,
            'elapsed_time_hours': float,
            'time_step_hours': float,
            'grid_power_mw': float,
            'solar_power_mw': float,
            'discharge_power_mw': float,
            'energy_before_mwh': float,
            'energy_after_mwh': float,
            'energy_in_from_grid_mwh': float,
            'energy_in_from_solar_mwh': float,
            'energy_in_total_mwh': float,
            'energy_out_total_mwh': float,
            'loss_charging_inefficiency_mwh': float,
            'loss_discharge_inefficiency_mwh': float,
            'loss_auxiliary_power_mwh': float,
            'loss_self_discharge_mwh': float,
            'loss_total_mwh': float,
            'timestamp_iso': str (optional)
        }

    Raises:
        ValueError: If logs list is empty
        IOError: If unable to write to file
    """
    if not logs:
        raise ValueError("Cannot write empty logs to CSV")

    # Define column order for CSV output
    base_columns = [
        "simulation_step",
        "elapsed_time_hours",
        "time_step_hours",
        "grid_power_mw",
        "solar_power_mw",
        "discharge_power_mw",
        "energy_before_mwh",
        "energy_after_mwh",
        "energy_in_from_grid_mwh",
        "energy_in_from_solar_mwh",
        "energy_in_total_mwh",
        "energy_out_total_mwh",
        "loss_charging_inefficiency_mwh",
        "loss_discharge_inefficiency_mwh",
        "loss_auxiliary_power_mwh",
        "loss_self_discharge_mwh",
        "loss_total_mwh",
    ]

    # Add timestamp column if present in logs
    fieldnames = base_columns.copy()
    if "timestamp_iso" in logs[0]:
        fieldnames.insert(2, "timestamp_iso")  # Insert after elapsed_time_hours

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(logs)
    except OSError as e:
        raise OSError(f"Unable to write CSV file: {e}") from e


def create_log_entry(
    step: int,
    t_hours: float,
    dt_hours: float,
    p_grid_mw: float,
    p_sol_mw: float,
    p_dis_mw: float,
    e_prev_mwh: float,
    e_next_mwh: float,
    losses: LossBreakdown,
    eta_ch: float,
    timestamp_iso: str | None = None,
) -> dict[str, Any]:
    """Create a standardized log entry dictionary for CSV export.

    Args:
        step (int): Step number (starting from 0)
        t_hours (float): Elapsed time in hours
        dt_hours (float): Time step duration in hours
        p_grid_mw (float): Grid power in MW
        p_sol_mw (float): Solar power in MW
        p_dis_mw (float): Discharge power in MW
        e_prev_mwh (float): Energy before step in MWh
        e_next_mwh (float): Energy after step in MWh
        losses (LossBreakdown): LossBreakdown object from compute_losses
        eta_ch (float): Charging efficiency (for calculating energy in)
        timestamp_iso (str | None): Optional ISO timestamp string

    Returns:
        dict[str, Any]: Dictionary ready for CSV export via write_simulation_csv
    """
    # Calculate energy flows
    e_in_grid_mwh = max(0.0, p_grid_mw) * eta_ch * dt_hours
    e_in_sol_mwh = max(0.0, p_sol_mw) * eta_ch * dt_hours
    e_in_total_mwh = e_in_grid_mwh + e_in_sol_mwh

    # Energy out from discharge losses (internal battery energy spent)
    e_out_mwh = losses.loss_discharge_ineff_mwh + (max(0.0, p_dis_mw) * dt_hours)

    log_entry: dict[str, Any] = {
        "simulation_step": step,
        "elapsed_time_hours": round(t_hours, 6),
        "time_step_hours": dt_hours,
        "grid_power_mw": p_grid_mw,
        "solar_power_mw": p_sol_mw,
        "discharge_power_mw": p_dis_mw,
        "energy_before_mwh": round(e_prev_mwh, 6),
        "energy_after_mwh": round(e_next_mwh, 6),
        "energy_in_from_grid_mwh": round(e_in_grid_mwh, 6),
        "energy_in_from_solar_mwh": round(e_in_sol_mwh, 6),
        "energy_in_total_mwh": round(e_in_total_mwh, 6),
        "energy_out_total_mwh": round(e_out_mwh, 6),
        "loss_charging_inefficiency_mwh": round(losses.loss_charge_ineff_mwh, 6),
        "loss_discharge_inefficiency_mwh": round(losses.loss_discharge_ineff_mwh, 6),
        "loss_auxiliary_power_mwh": round(losses.loss_aux_mwh, 6),
        "loss_self_discharge_mwh": round(losses.loss_self_discharge_mwh, 6),
        "loss_total_mwh": round(losses.total_loss_mwh, 6),
    }

    # Add timestamp if provided
    if timestamp_iso is not None:
        log_entry["timestamp_iso"] = timestamp_iso

    return log_entry


# Note: simulate_energy function removed - simulations should be built
# using step_energy in external simulation modules
