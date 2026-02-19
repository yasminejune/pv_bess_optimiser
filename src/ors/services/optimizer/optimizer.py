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

# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSV = "../../../../tests/data/optimizer/bess_test_data_intraday_15min.csv"  # columns: timestamp, price_intraday, solar_MW
OUTPUT_CSV = "../../../../tests/data/optimizer/bess_solution_v2_15min_1day.csv"

DT = 0.25  # 15 minutes in hours
N_D = 24 / DT  # 96 periods in a day

# Power limits
P_CH_MAX = 100.0  # MW
P_DIS_MAX = 100.0  # MW
P_RATED = 100.0  # MW

# Efficiencies
ETA_CH = 0.97
ETA_DIS = 0.97
ETA_SOL_SELL = 0.97  # losses when selling solar directly

# Losses
A_AUX = 0.005  # 0.5% of rated power
P_AUX = A_AUX * P_RATED  # MW
R_SD = 0.0005  # per hour (0.05%/h)

# 6h capacity
E_CAP = P_RATED * 6.0  # 600 MWh
E_MIN = 0.10 * E_CAP  # 60
E_MAX = 0.90 * E_CAP  # 540
E0 = 0.50 * E_CAP  # 300
E_TERMINAL_MIN = 0.4 * E_CAP  # 240 MWh

MAX_CYCLES_PER_DAY = 3

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(INPUT_CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"])
T = len(df)
assert T == 96, f"Expected 96 timesteps for 1 day at 15-min. Got {T}."

price = {i + 1: float(df.loc[i, "price_intraday"]) for i in range(T)}
solar = {i + 1: float(df.loc[i, "solar_MW"]) for i in range(T)}

# -----------------------------
# MODEL
# -----------------------------
m = ConcreteModel()
m.T = RangeSet(1, T)

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
    return m.P_grid[t] <= P_CH_MAX * m.z_grid[t]


m.grid_charge_limit = Constraint(m.T, rule=grid_charge_limit)


def sol_charge_limit(m, t):
    return m.P_sol_bat[t] <= P_CH_MAX * m.z_solbat[t]


m.sol_charge_limit = Constraint(m.T, rule=sol_charge_limit)


def discharge_limit(m, t):
    return m.P_dis[t] <= P_DIS_MAX * m.z_dis[t]


m.discharge_limit = Constraint(m.T, rule=discharge_limit)


# 4) Energy bounds
def energy_bounds_rule(m, t):
    return (E_MIN, m.E[t], E_MAX)


m.energy_bounds = Constraint(m.T, rule=energy_bounds_rule)

# 5) Initial energy
m.E_init = Constraint(expr=m.E[1] == E0)


# 6) Energy balance for t>=2
def energy_balance_rule(m, t):
    if t == 1:
        return Constraint.Skip

    # Total charging power into battery (cannot mix grid + solar by construction, but sum is fine)
    p_ch = m.P_grid[t] + m.P_sol_bat[t]

    return m.E[t] == (
        m.E[t - 1]
        + ETA_CH * p_ch * DT
        - (m.P_dis[t] * DT) / ETA_DIS
        - P_AUX * DT
        - m.E[t - 1] * R_SD * DT
    )


m.energy_balance = Constraint(m.T, rule=energy_balance_rule)

# 7) Terminal constraint to avoid end-of-day dump
m.terminal = Constraint(expr=m.E[T] >= E_TERMINAL_MIN)

# -----------------------------
# CYCLE COUNTING (max 3 cycles/day)
# c_t = z_grid + z_solbat, d_t = z_dis
# q_t turns on with charging, stays on through idle, turns off when discharging
# A cycle is counted when a discharge segment starts AND q_{t-1}=1
# -----------------------------

# Initialize flag & discharge-start & cycle vars at t=1
m.q_init = Constraint(expr=m.q[1] == 0)
m.s_dis_init = Constraint(expr=m.s_dis[1] == 0)
m.cycle_init = Constraint(expr=m.cycle[1] == 0)


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
    if t == 1:
        return Constraint.Skip
    d_t = m.z_dis[t]
    return m.q[t] >= m.q[t - 1] - d_t


m.q_hold = Constraint(m.T, rule=q_hold_rule)


# q_t <= q_{t-1} + c_t (flag consistency)
def q_limit_rule(m, t):
    if t == 1:
        return Constraint.Skip
    c_t = m.z_grid[t] + m.z_solbat[t]
    return m.q[t] <= m.q[t - 1] + c_t


m.q_limit = Constraint(m.T, rule=q_limit_rule)


# Start of discharge: s_dis_t >= d_t - d_{t-1}
def s_dis_rule(m, t):
    if t == 1:
        return Constraint.Skip
    return m.s_dis[t] >= m.z_dis[t] - m.z_dis[t - 1]


m.s_dis_def = Constraint(m.T, rule=s_dis_rule)


# cycle_t = AND(s_dis_t, q_{t-1})
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


# Daily cycle limit (single day horizon)
def daily_cycles_rule(m, t):
    window_start = max(1, t - (N_D - 1))  # 96 periods back (inclusive of t)
    return sum(m.cycle[k] for k in range(window_start, t + 1)) <= MAX_CYCLES_PER_DAY


m.daily_cycles = Constraint(m.T, rule=daily_cycles_rule)

# -----------------------------
# OBJECTIVE
# max Σ p_t * (P_dis + η_sol_sell*P_sol_sell - P_grid)
# (Δt omitted: constant scaling)
# -----------------------------
m.obj = Objective(
    expr=sum(m.p[t] * (m.P_dis[t] + ETA_SOL_SELL * m.P_sol_sell[t] - m.P_grid[t]) for t in m.T),
    sense=maximize,
)

# -----------------------------
# SOLVE
# -----------------------------
solver = SolverFactory("highs")
res = solver.solve(m, tee=True)  # tee=True shows progress

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

out.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved: {OUTPUT_CSV}")
print(f"Total profit (Δt-scaled): {out['profit_step'].sum():.2f}")
print(f"Total cycles counted: {int(out['cycle'].sum())}")
print(f"Terminal energy E_T: {out['E_MWh'].iloc[-1]:.2f} MWh (min {E_TERMINAL_MIN:.2f})")
