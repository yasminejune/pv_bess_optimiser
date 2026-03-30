import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd
from pyomo.environ import (
    Binary,
    ConcreteModel,
    Constraint,
    NonNegativeReals,
    Objective,
    Param,
    RangeSet,
    Var,
    maximize,
)

if TYPE_CHECKING:
    from src.ors.domain.models.battery import BatterySpec

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "battery_to_optimization"))


def create_input_df(**_kwargs) -> pd.DataFrame:
    """Build the intraday input DataFrame for the optimizer.

    Defaults to the live forecast integration helper. Tests may monkeypatch
    this symbol directly to inject synthetic inputs.
    """
    from src.ors.services.optimizer.integration import create_input_df as live_create_input_df

    return live_create_input_df(**_kwargs)


# -----------------------------
# CONFIG
# -----------------------------
# Input/output file paths
INPUT_CSV = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "..",
    "tests",
    "data",
    "optimizer",
    "bess_test_data_intraday_15min.csv",
)  # columns: timestamp, price_intraday, solar_MW
OUTPUT_CSV = "tests/data/optimizer/bess_solution_v2_15min_1day.csv"
BATTERY_STEP_CSV = "src/ors/services/battery_to_optimization/battery_storage.csv"

# Start optimization
START = "00:00"


# -----------------------------
# BATTERY STATE UTILITIES
# -----------------------------
def get_cycles_used_today(battery_logs: list[dict] | None = None) -> int:
    """Calculate number of charge/discharge cycles already used today.

    Args:
        battery_logs (list[dict] | None): Optional list of battery log entries for today.
                     If None, defaults to 0.

    Returns:
        int: Number of cycles already used today
    """
    if not battery_logs:
        return 0

    # Count cycle events in logs
    return sum(1 for log in battery_logs if log.get("cycle", 0) > 0)


def extract_optimizer_initial_state(
    battery_state: None = None,  # Will be BatteryState when imported
    e_cap_mwh: float = 0.0,
) -> tuple[float, str, int]:
    """Extract initial state parameters for optimizer from BatteryState.

    Args:
        battery_state (None): Current battery state from battery status service.
                      If None, uses default (50% SOC, idle, 0 cycles).
        e_cap_mwh (float): Energy capacity in MWh, used to compute 50% SOC default.

    Returns:
        tuple[float, str, int]: Tuple of (initial_energy_mwh, initial_mode, cycles_used_today)
    """
    if battery_state is None:
        return e_cap_mwh * 0.5, "idle", 0

    initial_energy_mwh = getattr(battery_state, "energy_mwh", e_cap_mwh * 0.5)
    initial_mode = getattr(battery_state, "operating_mode", "idle")

    # Note: cycles_used_today would need to come from separate source
    # (e.g., daily battery logs) as it's not part of current state
    cycles_used_today = 0  # TODO: Implement from daily logs

    return initial_energy_mwh, initial_mode, cycles_used_today


# -----------------------------
# LOAD DATA
# -----------------------------
def load_inputs(
    input_csv: str,
    historic_csv: str,
    start: str = START,
    now_dt: datetime | None = None,
    **kwargs,  # Kwargs for the create_input_df function
) -> tuple[dict[int, float], dict[int, float], float, int, int, pd.DataFrame]:
    """Load intraday and historic price CSVs and return model inputs.

    Args:
        input_csv (str): Path to intraday price CSV file
        historic_csv (str): Path to historic price CSV file
        start (str): Start time string
        now_dt (datetime | None): Current datetime
        **kwargs: Additional keyword arguments for create_input_df function

    Returns:
        tuple[dict[int, float], dict[int, float], float, int, int, pd.DataFrame]: Tuple of (price, solar, p_30, cycles_used_today, t_boundary, df)
    """
    df = create_input_df(**kwargs).copy()
    df["generation_kw"] = df["generation_kw"] / 1000  # kW → MW (column stays named generation_kw)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    n_t = len(df)
    assert n_t == 96, f"Expected 96 timesteps for 1 day at 15-min. Got {n_t}."

    price = {i + 1: float(df.loc[i, "price_intraday"]) for i in range(n_t)}
    solar = {i + 1: float(df.loc[i, "generation_kw"]) for i in range(n_t)}

    df_hist_price = pd.read_csv(historic_csv)
    df_hist_price["timestamp"] = pd.to_datetime(df_hist_price["timestamp"], utc=True)
    assert (
        len(df_hist_price) >= 30 * 96
    ), f"Expected at least 30 days of historic price data. Got {len(df_hist_price)} rows."

    df_hist_price["date"] = df_hist_price["timestamp"].dt.date
    df_hist_price["time_h"] = (
        df_hist_price["timestamp"].dt.hour + df_hist_price["timestamp"].dt.minute / 60
    )

    unique_dates = sorted(df_hist_price["date"].unique())
    last_30_dates = set(unique_dates[-30:])

    start_h, start_m = map(int, start.split(":"))
    start_frac = start_h + start_m / 60
    end_frac = start_frac + 0.5  # 30 minutes = 0.5 hours

    mask = (
        df_hist_price["date"].isin(last_30_dates)
        & (df_hist_price["time_h"] >= start_frac)
        & (df_hist_price["time_h"] < end_frac)
    )
    p_30 = float(df_hist_price.loc[mask, "price"].mean())

    return price, solar, p_30


# -----------------------------
# MODEL
# -----------------------------
def build_model(
    price: dict[int, float],
    solar: dict[int, float],
    p_30: float,
    cycles_used_today: int,
    t_boundary: int,
    q_init: int = 0,
    z_dis_init: int = 0,
    verbose: bool = False,
    battery_spec: "BatterySpec | None" = None,
    time_step_hours: float | None = None,
    initial_energy_mwh: float | None = None,
) -> ConcreteModel:
    """Build and return the Pyomo MILP model without solving.

    Args:
        price (dict[int, float]): Price data indexed by time period
        solar (dict[int, float]): Solar generation data indexed by time period
        p_30 (float): Historical average price for penalty calculation
        cycles_used_today (int): Number of charge/discharge cycles already used today
        t_boundary (int): Boundary time period for optimization horizon
        q_init (int): Value of the charge-since-last-discharge flag immediately *before*
            period 1. Pass 1 if the battery had charged since its last discharge
            at the start of this horizon; 0 otherwise. Used to anchor the q-hold
            and cycle-AND constraints at t=1.
        z_dis_init (int): Whether discharge was active in the step immediately *before* period 1.
            Used to detect a start-of-discharge event at t=1.
        verbose (bool): Enable verbose output for debugging
        battery_spec (BatterySpec): Battery specification from the optimization config template.
        time_step_hours (float | None): Time step duration in hours.
            If provided, takes precedence over optimization config value.
        initial_energy_mwh (float | None): Initial battery energy state in MWh.
            If provided, takes precedence over battery_spec.current_energy_mwh.

    Returns:
        ConcreteModel: Configured Pyomo optimization model

    Raises:
        ValueError: If battery_spec is not provided.
    """
    if battery_spec is None:
        raise ValueError("battery_spec is required. Load it from the optimization config template.")

    p_ch_max = battery_spec.rated_power_mw
    p_dis_max = battery_spec.rated_power_mw
    eta_ch = battery_spec.charge_efficiency
    eta_dis = battery_spec.discharge_efficiency
    e_cap = battery_spec.energy_capacity_mwh
    e_min = battery_spec.energy_capacity_mwh * battery_spec.min_soc_percent / 100.0
    e_max = battery_spec.energy_capacity_mwh * battery_spec.max_soc_percent / 100.0
    p_aux = battery_spec.auxiliary_power_mw
    r_sd = battery_spec.self_discharge_rate_per_hour
    max_cycles_per_day = battery_spec.max_cycles_per_day
    dt = time_step_hours if time_step_hours is not None else 0.25

    if verbose:
        print(f"Info: Using battery config: {p_ch_max} MW / {e_cap} MWh")

    n_t = len(price)
    m = ConcreteModel()
    m.T = RangeSet(1, n_t)

    # Initial energy: use provided value, configured current state, or default to 50% SOC
    if initial_energy_mwh is not None:
        e0 = initial_energy_mwh
    elif (
        hasattr(battery_spec, "current_energy_mwh") and battery_spec.current_energy_mwh is not None
    ):
        e0 = battery_spec.current_energy_mwh
    else:
        e0 = e_cap * 0.5

    # Validate initial state bounds
    if e0 < e_min or e0 > e_max:
        print(
            f"Warning: Initial energy {e0:.2f} MWh is outside bounds [{e_min:.2f}, {e_max:.2f}] MWh"
        )
        e0 = max(e_min, min(e_max, e0))  # Clamp to valid range
        print(f"Info: Clamped to {e0:.2f} MWh ({e0/e_cap*100:.1f}% SOC)")

    m.p = Param(m.T, initialize=price)
    m.S = Param(m.T, initialize=solar)

    # -----------------------------
    # VARIABLES
    # -----------------------------
    # Power decisions
    m.P_grid = Var(m.T, domain=NonNegativeReals)  # MW (grid -> battery)
    m.P_dis = Var(m.T, domain=NonNegativeReals)  # MW (battery -> grid)

    m.P_sol_bat = Var(m.T, domain=NonNegativeReals)  # MW (solar -> battery)
    m.P_sol_sell = Var(m.T, domain=NonNegativeReals)  # MW (solar -> grid direct)

    # Battery mode binaries (mutually exclusive)
    m.z_grid = Var(m.T, domain=Binary)  # battery charges from grid
    m.z_solbat = Var(m.T, domain=Binary)  # battery charges from solar
    m.z_dis = Var(m.T, domain=Binary)  # battery discharges

    # State of energy
    m.E = Var(m.T)  # MWh (bounded by constraints)

    # Cycle logic
    m.q = Var(m.T, domain=Binary)  # charge-since-last-discharge flag
    m.s_dis = Var(m.T, domain=Binary)  # start-of-discharge indicator
    m.cycle = Var(m.T, domain=Binary)  # cycle event

    # -----------------------------
    # CONSTRAINTS
    # -----------------------------

    # 1) Battery mode exclusivity: at most one of {grid charge, solar charge, discharge}
    def mode_excl_rule(m, t):
        return m.z_grid[t] + m.z_solbat[t] + m.z_dis[t] <= 1

    m.mode_excl = Constraint(m.T, rule=mode_excl_rule)

    # 2) Solar balance (no curtailment): all solar must be used
    def solar_balance_rule(m, t):
        return m.P_sol_bat[t] + m.P_sol_sell[t] == m.S[t]

    m.solar_balance = Constraint(m.T, rule=solar_balance_rule)

    # 3) Power activation + limits
    def grid_charge_limit(m, t):
        return m.P_grid[t] <= p_ch_max * m.z_grid[t]

    m.grid_charge_limit = Constraint(m.T, rule=grid_charge_limit)

    def sol_charge_limit(m, t):
        return m.P_sol_bat[t] <= p_ch_max * m.z_solbat[t]

    m.sol_charge_limit = Constraint(m.T, rule=sol_charge_limit)

    def discharge_limit(m, t):
        return m.P_dis[t] <= p_dis_max * m.z_dis[t]

    m.discharge_limit = Constraint(m.T, rule=discharge_limit)

    # 4) Energy bounds
    def energy_bounds_rule(m, t):
        return (e_min, m.E[t], e_max)

    m.energy_bounds = Constraint(m.T, rule=energy_bounds_rule)

    # 5) Energy balance for all t (t=1 uses E0 as the prior state)
    # Previously E_init fixed E[1]=E0 and the balance skipped t=1, which let
    # P_dis[1] earn revenue without reducing any stored energy.  Including t=1
    # here (with E0 as the "previous" state) closes that loophole.
    def energy_balance_rule(m, t):
        e_prev = e0 if t == 1 else m.E[t - 1]
        p_ch = m.P_grid[t] + m.P_sol_bat[t]

        return m.E[t] == (
            e_prev
            + eta_ch * p_ch * dt
            - (m.P_dis[t] * dt) / eta_dis
            - p_aux * dt
            - e_prev * r_sd * dt
        )

    m.energy_balance = Constraint(m.T, rule=energy_balance_rule)

    # -----------------------------
    # CYCLE COUNTING (max 3 cycles/day)
    # c_t = z_grid + z_solbat, d_t = z_dis
    # q_t turns on with charging, stays on through idle, turns off when discharging
    # A cycle is counted when a discharge segment starts AND q_{t-1}=1
    # -----------------------------
    # m.q_init = Constraint(expr=m.q[1] == 0)
    # m.s_dis_init = Constraint(expr=m.s_dis[1] == 0)
    # m.cycle_init = Constraint(expr=m.cycle[1] == 0)

    # q_t >= c_t (charging turns flag ON)
    def q_on_rule(m, t):
        c_t = m.z_grid[t] + m.z_solbat[t]
        return m.q[t] >= c_t

    m.q_on = Constraint(m.T, rule=q_on_rule)

    # q_t <= 1 - d_t (discharging turns flag OFF)
    def q_off_rule(m, t):
        d_t = m.z_dis[t]
        return m.q[t] <= 1 - d_t

    m.q_off = Constraint(m.T, rule=q_off_rule)

    # q_t >= q_{t-1} - d_t (idle does not reset)
    def q_hold_rule(m, t):
        q_prev = q_init if t == 1 else m.q[t - 1]
        return m.q[t] >= q_prev - m.z_dis[t]

    m.q_hold = Constraint(m.T, rule=q_hold_rule)

    # q_t <= q_{t-1} + c_t (flag consistency)
    def q_limit_rule(m, t):
        q_prev = q_init if t == 1 else m.q[t - 1]
        c_t = m.z_grid[t] + m.z_solbat[t]
        return m.q[t] <= q_prev + c_t

    m.q_limit = Constraint(m.T, rule=q_limit_rule)

    # Start of discharge: s_dis_t >= d_t - d_{t-1}
    def s_dis_rule(m, t):
        z_dis_prev = z_dis_init if t == 1 else m.z_dis[t - 1]
        return m.s_dis[t] >= m.z_dis[t] - z_dis_prev

    m.s_dis_def = Constraint(m.T, rule=s_dis_rule)

    # cycle_t = AND(s_dis_t, q_{t-1})
    def cycle_and1(m, t):
        return m.cycle[t] <= m.s_dis[t]

    def cycle_and2(m, t):
        q_prev = q_init if t == 1 else m.q[t - 1]
        return m.cycle[t] <= q_prev

    def cycle_and3(m, t):
        q_prev = q_init if t == 1 else m.q[t - 1]
        return m.cycle[t] >= m.s_dis[t] + q_prev - 1

    m.cycle_and1 = Constraint(m.T, rule=cycle_and1)
    m.cycle_and2 = Constraint(m.T, rule=cycle_and2)
    m.cycle_and3 = Constraint(m.T, rule=cycle_and3)

    # Daily cycle limit (single day horizon, accounting for cycles already used)
    def daily_cycles_rule(m, t):
        n_d = 24 / dt  # periods in a day based on actual time step
        window_start = max(1, t - (n_d - 1))  # periods back (inclusive of t)
        cycles_in_optimization = sum(m.cycle[k] for k in range(window_start, t + 1))
        return cycles_in_optimization <= max_cycles_per_day - cycles_used_today

    m.daily_cycles = Constraint(m.T, rule=daily_cycles_rule)

    # Terminal energy constraint (relaxed for debugging)
    def terminal_energy_rule(m, t):
        if t == n_t:
            return m.E[t] >= e_min  # Use minimum instead of terminal minimum
        return Constraint.Skip

    m.terminal_energy = Constraint(m.T, rule=terminal_energy_rule)
    if verbose:
        print(f"Info: Terminal energy constraint: E[{n_t}] >= {e_min:.2f} MWh (relaxed)")

    # -----------------------------
    # OBJECTIVE
    # max Σ p_t·Δt·(P_dis + η_sol_sell·P_sol_sell - P_grid) + (E_T - E_min)·p_30
    # All terms in £ for consistent weighting of dispatch vs terminal value
    # -----------------------------
    eta_sol_sell = 0.97  # Solar sell efficiency (keep as constant)
    m.obj = Objective(
        expr=sum(
            m.p[t] * (m.P_dis[t] + eta_sol_sell * m.P_sol_sell[t] - m.P_grid[t]) * dt for t in m.T
        )
        + (m.E[n_t] - e_min) * p_30 * eta_dis,
        sense=maximize,
    )

    return m
