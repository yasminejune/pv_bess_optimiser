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
    SolverFactory,
    Var,
    maximize,
    value,
)

if TYPE_CHECKING:
    from src.ors.domain.models.battery import BatterySpec

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "battery_to_optimization"))
from battery_inference import (
    load_optimizer_battery_config,
)

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

# Load battery configuration from battery module
# Module-level configuration loading - prints moved to runtime context
_battery_config_loaded = False
_load_error = None

try:
    battery_params, sim_defaults = load_optimizer_battery_config()
    _battery_config_loaded = True
except Exception as e:
    _load_error = e
    # Fallback to original hardcoded values
    class MockBatteryParams:
        p_rated_mw = 100.0
        eta_ch = 0.97
        eta_dis = 0.97
        e_cap_mwh = 600.0
        e_min_mwh = 60.0
        e_max_mwh = 540.0
        p_aux_mw = 0.5
        r_sd_per_hour = 0.0005

    battery_params = MockBatteryParams()
    sim_defaults = {"dt_hours": 0.25, "enforce_bounds": True}

# Extract parameters from battery config
DT = sim_defaults["dt_hours"]  # 15 minutes in hours
N_D = 24 / DT  # 96 periods in a day

# Power limits
P_CH_MAX = battery_params.p_rated_mw  # MW
P_DIS_MAX = battery_params.p_rated_mw  # MW
P_RATED = battery_params.p_rated_mw  # MW

# Efficiencies
ETA_CH = battery_params.eta_ch
ETA_DIS = battery_params.eta_dis
ETA_SOL_SELL = 0.97  # losses when selling solar directly

# Losses (from battery params)
P_AUX = battery_params.p_aux_mw  # MW
R_SD = battery_params.r_sd_per_hour  # per hour

# Energy capacity and bounds
E_CAP = battery_params.e_cap_mwh  # MWh
E_MIN = battery_params.e_min_mwh  # MWh
E_MAX = battery_params.e_max_mwh  # MWh
# E0 is the initial energy state - can be overridden by backtesting or other callers
E0 = E_CAP * 0.5  # Default to 50% SOC
E_TERMINAL_MIN = E_CAP * 0.4  # 40% minimum terminal SOC

MAX_CYCLES_PER_DAY = 3

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
) -> tuple[float, str, int]:
    """Extract initial state parameters for optimizer from BatteryState.

    Args:
        battery_state (None): Current battery state from battery status service.
                      If None, uses default (50% SOC, idle, 0 cycles).

    Returns:
        tuple[float, str, int]: Tuple of (initial_energy_mwh, initial_mode, cycles_used_today)
    """
    if battery_state is None:
        # Default fallback when no battery state available
        return E_CAP * 0.5, "idle", 0

    # Extract from battery state (fields would match BatteryState model)
    initial_energy_mwh = getattr(battery_state, "energy_mwh", E_CAP * 0.5)
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
    df["generation_kw"] = df["generation_kw"]/1000
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    n_t = len(df)
    assert n_t == 96, f"Expected 96 timesteps for 1 day at 15-min. Got {n_t}."

    price = {i + 1: float(df.loc[i, "price_intraday"]) for i in range(n_t)}
    solar = {i + 1: float(df.loc[i, "solar_MW"]) for i in range(n_t)}

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
        battery_spec (BatterySpec | None): Battery specification to use for optimization.
            If provided, takes precedence over internal battery config.
        time_step_hours (float | None): Time step duration in hours.
            If provided, takes precedence over internal config.

    Returns:
        ConcreteModel: Configured Pyomo optimization model
    """
    
    # Determine battery parameters source
    if battery_spec is not None:
        # Use provided battery specification (main config)
        p_ch_max = battery_spec.rated_power_mw
        p_dis_max = battery_spec.rated_power_mw
        eta_ch = battery_spec.charge_efficiency
        eta_dis = battery_spec.discharge_efficiency
        e_cap = battery_spec.energy_capacity_mwh
        e_min = battery_spec.energy_capacity_mwh * battery_spec.min_soc_percent / 100.0
        e_max = battery_spec.energy_capacity_mwh * battery_spec.max_soc_percent / 100.0
        p_aux = battery_spec.auxiliary_power_mw
        r_sd = battery_spec.self_discharge_rate_per_hour
        
        # Use provided time step or default
        dt = time_step_hours if time_step_hours is not None else 0.25  # 15 minutes default
        
        if verbose:
            print(f"Info: Using provided battery config: {p_ch_max} MW / {e_cap} MWh")
    else:
        # Fall back to internal battery config for backward compatibility
        global _battery_config_loaded, _load_error
        if not _battery_config_loaded and _load_error:
            if verbose:
                print(f"Warning: Could not load battery config: {_load_error}")
                print("Info: Using default parameters...")
        elif _battery_config_loaded and verbose:
            print(f"Info: Using internal battery config: {battery_params.p_rated_mw} MW / {battery_params.e_cap_mwh} MWh")
        
        # Use global variables
        p_ch_max = P_CH_MAX
        p_dis_max = P_DIS_MAX
        eta_ch = ETA_CH
        eta_dis = ETA_DIS
        e_cap = E_CAP
        e_min = E_MIN
        e_max = E_MAX
        p_aux = P_AUX
        r_sd = R_SD
        dt = time_step_hours if time_step_hours is not None else DT
    
    n_t = len(price)
    m = ConcreteModel()
    m.T = RangeSet(1, n_t)

    # Set initial energy state from module-level E0 variable
    e0 = E0

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
    #m.q_init = Constraint(expr=m.q[1] == 0)
    #m.s_dis_init = Constraint(expr=m.s_dis[1] == 0)
    #m.cycle_init = Constraint(expr=m.cycle[1] == 0)

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
        return cycles_in_optimization <= MAX_CYCLES_PER_DAY - cycles_used_today

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


if __name__ == "__main__":
    # Load price and solar data
    price, solar, p_30 = load_inputs(INPUT_CSV, "dummy.csv")

    # Extract current battery state (in real usage, this would come from battery_status service)
    # For demonstration, show both default and custom initial state
    print("Info: Battery State Options:")
    print("  1. Default state (50% SOC, idle, 0 cycles)")
    print("  2. Custom state (for demonstration)")

    # Option 1: Default state (current behavior maintained)
    print("\nInfo: Using default battery state...")
    initial_energy_mwh, initial_mode, cycles_used_today = extract_optimizer_initial_state(None)
    print(
        f"  Initial energy: {initial_energy_mwh:.1f} MWh ({initial_energy_mwh/E_CAP*100:.1f}% SOC)"
    )
    print(f"  Initial mode: {initial_mode}")
    print(f"  Cycles used today: {cycles_used_today}")

    m = build_model(
        price,
        solar,
        p_30,
        initial_energy_mwh=initial_energy_mwh,
        initial_mode=initial_mode,
        cycles_used_today=cycles_used_today,
    )

    # -----------------------------
    # SOLVE
    # -----------------------------
    solver = SolverFactory("highs")
    res = solver.solve(m, tee=True)  # tee=True shows progress

    df = pd.read_csv(INPUT_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # -----------------------------
    # EXPORT RESULTS
    # -----------------------------
    out = df.copy()

    out["P_grid_MW"] = [value(m.P_grid[t]) for t in m.T]
    out["P_dis_MW"] = [value(m.P_dis[t]) for t in m.T]
    out["P_sol_bat_MW"] = [value(m.P_sol_bat[t]) for t in m.T]
    out["P_sol_sell_MW"] = [value(m.P_sol_sell[t]) for t in m.T]
    out["E_MWh"] = [value(m.E[t]) for t in m.T]

    out["z_grid"] = [int(round(value(m.z_grid[t]))) for t in m.T]
    out["z_solbat"] = [int(round(value(m.z_solbat[t]))) for t in m.T]
    out["z_dis"] = [int(round(value(m.z_dis[t]))) for t in m.T]

    out["q_flag"] = [int(round(value(m.q[t]))) for t in m.T]
    out["s_dis"] = [int(round(value(m.s_dis[t]))) for t in m.T]
    out["cycle"] = [int(round(value(m.cycle[t]))) for t in m.T]

    # Profit per step (with Δt scaling to reflect energy over interval)
    out["profit_step"] = (
        out["price_intraday"]
        * (out["P_dis_MW"] + ETA_SOL_SELL * out["P_sol_sell_MW"] - out["P_grid_MW"])
        * DT
    )

    # Save results and display summary
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Success: Results saved to {OUTPUT_CSV}")
    print(f"Info: Total profit: €{out['profit_step'].sum():.2f}")
    print(f"Info: Cycles used: {int(out['cycle'].sum())}")
