from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

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

# Battery integration functions (must be a proper package import)
from ors.services.battery_to_optimization.battery_inference import (
    create_enhanced_optimizer_output,
    load_optimizer_battery_config,  # imported in case you want direct use
)
from ors.services.optimizer.integration import create_input_df

# -----------------------------
# PATHS / CONFIG
# -----------------------------
HERE = Path(__file__).resolve()
# optimizer.py should be at: src/ors/services/optimizer/optimizer.py
# parents:
# 0=optimizer.py, 1=optimizer, 2=services, 3=ors, 4=src, 5=repo_root
REPO_ROOT = HERE.parents[4]

INPUT_CSV = (
    REPO_ROOT / "tests/data/optimizer/bess_test_data_intraday_15min.csv"
)  # columns: timestamp, price_intraday, solar_MW
OUTPUT_CSV = REPO_ROOT / "tests/data/optimizer/bess_solution_v2_15min_1day.csv"
HISTORIC_PRICE_CSV = REPO_ROOT / "tests/data/prediction/price_data.csv"
BATTERY_STEP_CSV = REPO_ROOT / "src/ors/services/battery_to_optimization/battery_storage.csv"

# Start optimization horizon reference (used for discharge-window average)
START = "00:00"

# -----------------------------
# LOAD BATTERY CONFIG
# -----------------------------
print("Loading battery configuration...")
try:
    battery_params, sim_defaults = load_optimizer_battery_config()
    print(f"✓ Loaded battery config: {battery_params.p_rated_mw}MW / {battery_params.e_cap_mwh}MWh")
except Exception as e:
    print(f"⚠ Could not load battery config: {e}")
    print("Using default parameters...")

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
DT = float(sim_defaults["dt_hours"])  # 15 minutes = 0.25 hours
N_D = int(round(24 / DT))  # 96 periods in a day

# Power limits
P_CH_MAX = float(battery_params.p_rated_mw)  # MW
P_DIS_MAX = float(battery_params.p_rated_mw)  # MW
P_RATED = float(battery_params.p_rated_mw)  # MW

# Efficiencies
ETA_CH = float(battery_params.eta_ch)
ETA_DIS = float(battery_params.eta_dis)
ETA_SOL_SELL = 0.97  # losses when selling solar directly

# Losses
P_AUX = float(battery_params.p_aux_mw)  # MW
R_SD = float(battery_params.r_sd_per_hour)  # per hour

# Energy capacity and bounds
E_CAP = float(battery_params.e_cap_mwh)  # MWh
E_MIN = float(battery_params.e_min_mwh)  # MWh
E_MAX = float(battery_params.e_max_mwh)  # MWh
E0 = 0.50 * E_CAP  # Start at 50% SOC

# IMPORTANT: H_30 is used inside load_inputs() to compute the historic-price window
H_30 = (E_MAX - E_MIN) / P_DIS_MAX  # hours

MAX_CYCLES_PER_DAY = 3


# -----------------------------
# LOAD DATA
# -----------------------------
def load_inputs(
    input_csv: str,
    historic_csv: str,
    output_csv: str | None = None,
    start: str = START,
    now_dt: datetime | None = None,
    **kwargs,  # Kwargs for the create_input_df function
) -> tuple[dict[int, float], dict[int, float], float, int, int]:
    """Load intraday and historic price CSVs and return model inputs."""
    df = create_input_df(**kwargs)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    n_t = len(df)
    assert n_t == N_D, f"Expected {N_D} timesteps for 1 day at {DT}-hour steps. Got {n_t}."

    price = {i + 1: float(df.loc[i, "price_intraday"]) for i in range(n_t)}
    solar = {i + 1: float(df.loc[i, "generation_kw"]) for i in range(n_t)}

    df_hist_price = pd.read_csv(historic_csv)
    df_hist_price["timestamp"] = pd.to_datetime(df_hist_price["timestamp"], utc=True)

    min_rows = 30 * N_D
    assert (
        len(df_hist_price) >= min_rows
    ), f"Expected at least 30 days of historic price data ({min_rows} rows). Got {len(df_hist_price)} rows."

    df_hist_price["date"] = df_hist_price["timestamp"].dt.date
    df_hist_price["time_h"] = (
        df_hist_price["timestamp"].dt.hour + df_hist_price["timestamp"].dt.minute / 60
    )

    unique_dates = sorted(df_hist_price["date"].unique())
    last_30_dates = set(unique_dates[-30:])

    start_h, start_m = map(int, start.split(":"))
    start_frac = start_h + start_m / 60
    end_frac = start_frac + H_30

    mask = (
        df_hist_price["date"].isin(last_30_dates)
        & (df_hist_price["time_h"] >= start_frac)
        & (df_hist_price["time_h"] < end_frac)
    )
    p_30 = float(df_hist_price.loc[mask, "price"].mean())

    # Count cycles already used in the current 23:00-23:00 day
    now = now_dt or datetime.now()
    if now.hour >= 23:
        day_start = now.replace(hour=23, minute=0, second=0, microsecond=0)
    else:
        day_start = (now - timedelta(days=1)).replace(hour=23, minute=0, second=0, microsecond=0)

    if output_csv is not None:
        df_output = pd.read_csv(output_csv)
        df_output["timestamp"] = pd.to_datetime(df_output["timestamp"])
        cycles_used_today = int(df_output[df_output["timestamp"] >= day_start]["cycle"].sum())
    else:
        cycles_used_today = 0

    # Period index (1-based) of the next 23:00 within the horizon; clamped to n_t
    next_23 = day_start + timedelta(days=1)
    hours_until_23 = (next_23 - now).total_seconds() / 3600
    t_boundary = min(n_t, int(round(hours_until_23 / DT)))

    return price, solar, p_30, cycles_used_today, t_boundary


# -----------------------------
# MODEL
# -----------------------------
def build_model(
    price: dict[int, float],
    solar: dict[int, float],
    p_30: float,
    cycles_used_today: int,
    t_boundary: int,
) -> ConcreteModel:
    """Build and return the Pyomo MILP model without solving."""
    n_t = len(price)
    m = ConcreteModel()
    m.T = RangeSet(1, n_t)

    m.p = Param(m.T, initialize=price)  # £/MWh
    m.S = Param(m.T, initialize=solar)  # MW

    # -----------------------------
    # VARIABLES
    # -----------------------------
    m.P_grid = Var(m.T, domain=NonNegativeReals)  # MW (grid -> battery)
    m.P_dis = Var(m.T, domain=NonNegativeReals)  # MW (battery -> grid)

    m.P_sol_bat = Var(m.T, domain=NonNegativeReals)  # MW (solar -> battery)
    m.P_sol_sell = Var(m.T, domain=NonNegativeReals)  # MW (solar -> grid direct)

    m.z_grid = Var(m.T, domain=Binary)
    m.z_solbat = Var(m.T, domain=Binary)
    m.z_dis = Var(m.T, domain=Binary)

    m.E = Var(m.T)  # MWh (bounded by constraints)

    # Cycle logic
    m.q = Var(m.T, domain=Binary)  # charge-since-last-discharge flag
    m.s_dis = Var(m.T, domain=Binary)  # start-of-discharge indicator
    m.cycle = Var(m.T, domain=Binary)  # cycle event

    # -----------------------------
    # CONSTRAINTS
    # -----------------------------
    def mode_excl_rule(m, t):
        return m.z_grid[t] + m.z_solbat[t] + m.z_dis[t] <= 1

    m.mode_excl = Constraint(m.T, rule=mode_excl_rule)

    def solar_balance_rule(m, t):
        return m.P_sol_bat[t] + m.P_sol_sell[t] == m.S[t]

    m.solar_balance = Constraint(m.T, rule=solar_balance_rule)

    def grid_charge_limit(m, t):
        return m.P_grid[t] <= P_CH_MAX * m.z_grid[t]

    m.grid_charge_limit = Constraint(m.T, rule=grid_charge_limit)

    def sol_charge_limit(m, t):
        return m.P_sol_bat[t] <= P_CH_MAX * m.z_solbat[t]

    m.sol_charge_limit = Constraint(m.T, rule=sol_charge_limit)

    def discharge_limit(m, t):
        return m.P_dis[t] <= P_DIS_MAX * m.z_dis[t]

    m.discharge_limit = Constraint(m.T, rule=discharge_limit)

    def energy_bounds_rule(m, t):
        return (E_MIN, m.E[t], E_MAX)

    m.energy_bounds = Constraint(m.T, rule=energy_bounds_rule)

    # Energy balance for all t (including t=1 using E0 as previous state)
    def energy_balance_rule(m, t):
        e_prev = E0 if t == 1 else m.E[t - 1]
        p_ch = m.P_grid[t] + m.P_sol_bat[t]
        return m.E[t] == (
            e_prev
            + ETA_CH * p_ch * DT
            - (m.P_dis[t] * DT) / ETA_DIS
            - P_AUX * DT
            - e_prev * R_SD * DT
        )

    m.energy_balance = Constraint(m.T, rule=energy_balance_rule)

    # -----------------------------
    # CYCLE COUNTING (max 3 cycles/day)
    # -----------------------------
    m.q_init = Constraint(expr=m.q[1] == 0)
    m.s_dis_init = Constraint(expr=m.s_dis[1] == 0)
    m.cycle_init = Constraint(expr=m.cycle[1] == 0)

    def q_on_rule(m, t):
        c_t = m.z_grid[t] + m.z_solbat[t]
        return m.q[t] >= c_t

    m.q_on = Constraint(m.T, rule=q_on_rule)

    def q_off_rule(m, t):
        d_t = m.z_dis[t]
        return m.q[t] <= 1 - d_t

    m.q_off = Constraint(m.T, rule=q_off_rule)

    def q_hold_rule(m, t):
        if t == 1:
            return Constraint.Skip
        d_t = m.z_dis[t]
        return m.q[t] >= m.q[t - 1] - d_t

    m.q_hold = Constraint(m.T, rule=q_hold_rule)

    def q_limit_rule(m, t):
        if t == 1:
            return Constraint.Skip
        c_t = m.z_grid[t] + m.z_solbat[t]
        return m.q[t] <= m.q[t - 1] + c_t

    m.q_limit = Constraint(m.T, rule=q_limit_rule)

    def s_dis_rule(m, t):
        if t == 1:
            return Constraint.Skip
        return m.s_dis[t] >= m.z_dis[t] - m.z_dis[t - 1]

    m.s_dis_def = Constraint(m.T, rule=s_dis_rule)

    def cycle_and1(m, t):
        if t == 1:
            return Constraint.Skip
        return m.cycle[t] <= m.s_dis[t]

    def cycle_and2(m, t):
        if t == 1:
            return Constraint.Skip
        return m.cycle[t] <= m.q[t - 1]

    def cycle_and3(m, t):
        if t == 1:
            return Constraint.Skip
        return m.cycle[t] >= m.s_dis[t] + m.q[t - 1] - 1

    m.cycle_and1 = Constraint(m.T, rule=cycle_and1)
    m.cycle_and2 = Constraint(m.T, rule=cycle_and2)
    m.cycle_and3 = Constraint(m.T, rule=cycle_and3)

    # Constraint 1 — 23:00-to-23:00 calendar-day cap.
    # Sum of cycles in the current-day portion of the horizon (periods 1..t_boundary), plus
    # cycles already executed since the last 23:00, must not exceed the daily limit.
    m.cycles_cur_day = Constraint(
        expr=sum(m.cycle[k] for k in range(1, t_boundary + 1))
        <= MAX_CYCLES_PER_DAY - cycles_used_today
    )

    # Constraint 2 — rolling 24h forward cap.
    # Total cycles planned across the full 96-period horizon must not exceed the daily limit.
    # Together with cycles_cur_day this ensures no consecutive 24h window ever exceeds the cap:
    # any cycles used before 23:00 reduce the budget available after 23:00 in the same horizon.
    m.daily_cycles = Constraint(expr=sum(m.cycle[k] for k in m.T) <= MAX_CYCLES_PER_DAY)

    # -----------------------------
    # OBJECTIVE (Δt-scaled to convert MW to MWh over interval)
    # max Σ p_t·Δt·(P_dis + η_sol_sell·P_sol_sell - P_grid) + terminal value
    # -----------------------------
    m.obj = Objective(
        expr=sum(
            m.p[t] * (m.P_dis[t] + ETA_SOL_SELL * m.P_sol_sell[t] - m.P_grid[t]) * DT for t in m.T
        )
        + (m.E[n_t] - E_MIN) * p_30 * ETA_DIS,
        sense=maximize,
    )

    return m


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    price, solar, p_30, cycles_used_today, t_boundary = load_inputs(
        str(INPUT_CSV), str(HISTORIC_PRICE_CSV), str(OUTPUT_CSV)
    )
    m = build_model(price, solar, p_30, cycles_used_today, t_boundary)

    # Solve (try HiGHS first)
    print("\n🔄 Solving optimization model...")
    solvers_to_try = ["highs", "glpk", "cbc", "gurobi", "cplex"]
    solver = None
    for solver_name in solvers_to_try:
        try:
            test_solver = SolverFactory(solver_name)
            if test_solver.available():
                solver = test_solver
                print(f"✓ Using solver: {solver_name}")
                break
        except Exception:
            continue

    if solver is None:
        raise RuntimeError(
            "No optimization solver available. Install one of: highs (highspy), glpk, cbc, gurobi, cplex."
        )

    res = solver.solve(m, tee=True)

    # Load original input for timestamps
    df = create_input_df()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Export results
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

    out["profit_step"] = (
        out["price_intraday"]
        * (out["P_dis_MW"] + ETA_SOL_SELL * out["P_sol_sell_MW"] - out["P_grid_MW"])
        * DT
    )

    out.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✓ Saved: {OUTPUT_CSV}")
    print(f"Total profit (Δt-scaled): {out['profit_step'].sum():.2f}")
    print(f"Total cycles counted: {int(out['cycle'].sum())}")
    print(f"Terminal energy E_T: {out['E_MWh'].iloc[-1]:.2f} MWh")
    print(f"Terminal value (p_30={p_30:.2f}): {(out['E_MWh'].iloc[-1] - E_MIN) * p_30:.2f}")

    # Optional enhanced battery module output
    print("\n🔋 Creating enhanced battery module output...")
    try:
        start_datetime: datetime = out["timestamp"].iloc[0].to_pydatetime()

        export_results = create_enhanced_optimizer_output(
            df_results=out,
            csv_path=str(BATTERY_STEP_CSV),
            params=battery_params,
            dt_hours=DT,
            start_datetime=start_datetime,
            validate=True,
        )

        print(f"✓ Battery logs saved: {BATTERY_STEP_CSV}")
        if isinstance(export_results, dict) and "validation" in export_results:
            validation = export_results["validation"]
            if validation.get("is_valid"):
                print("✓ Energy balance validation: PASSED")
                print(f"  Max error: {validation.get('max_error', 0.0):.6f} MWh")
            else:
                print("⚠ Energy balance validation: FAILED")
                print(f"  Max error: {validation.get('max_error', 0.0):.6f} MWh")

    except Exception as e:
        print(f"⚠ Enhanced output failed: {e}")
        print("Original CSV output is still available.")
