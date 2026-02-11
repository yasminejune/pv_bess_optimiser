import pandas as pd
from pyomo.environ import (
    ConcreteModel, RangeSet, Param, Var, Constraint, Objective, maximize,
    NonNegativeReals, Binary, value, SolverFactory
)

# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSV  = "bess_test_data_intraday_15min.csv"   
OUTPUT_CSV = "bess_solution_15min_1day.csv"

DT = 0.25  # 15 min in hours

# Battery / prototype params (your current assumptions)
P_CH_MAX = 100.0  # MW
P_DIS_MAX = 100.0 # MW
ETA_CH = 0.97
ETA_DIS = 0.97
P_RATED = 100.0
A_AUX = 0.005  # 0.5%
P_AUX = A_AUX * P_RATED  # MW
R_SD = 0.0005  # per hour (0.05%/h)

# capacity
E_CAP = P_RATED * 3.0  
E_MIN = 0.10 * E_CAP  
E_MAX = 0.90 * E_CAP   
E0    = 0.50 * E_CAP   

MAX_CYCLES_PER_DAY = 3

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv(INPUT_CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"])
T = len(df)


# -----------------------------
# MODEL
# -----------------------------
m = ConcreteModel()
m.T = RangeSet(1, T)

# Parameters indexed by t
price_dict = {i+1: float(df.loc[i, "price_intraday"]) for i in range(T)}
solar_dict = {i+1: float(df.loc[i, "solar_MW"]) for i in range(T)}

m.p = Param(m.T, initialize=price_dict)
m.S = Param(m.T, initialize=solar_dict)

# Decision variables
m.P_grid = Var(m.T, domain=NonNegativeReals)  # MW
m.P_dis  = Var(m.T, domain=NonNegativeReals)  # MW

# Mode binaries (solar charge / grid charge / discharge). Idle => all 0.
m.z_sol  = Var(m.T, domain=Binary)
m.z_grid = Var(m.T, domain=Binary)
m.z_dis  = Var(m.T, domain=Binary)

# State variable (energy)
m.E = Var(m.T)  # MWh (bounded via constraints)

# Solar-to-battery (linear because S_t is constant and z_sol is binary)
m.P_sol = Var(m.T, domain=NonNegativeReals)  # MW


# Cycle counting logic variables
m.q = Var(m.T, domain=Binary)        # "charged since last discharge" flag
m.s_dis = Var(m.T, domain=Binary)    # start of discharge segment
m.cycle = Var(m.T, domain=Binary)    # cycle event

# -----------------------------
# CONSTRAINTS
# -----------------------------

# (1) Mode exclusivity: only one action per interval (or none)
def mode_excl_rule(m, t):
    return m.z_sol[t] + m.z_grid[t] + m.z_dis[t] <= 1
m.mode_excl = Constraint(m.T, rule=mode_excl_rule)

# (2) Solar routing when solar mode is on: P_sol = S * z_sol
def solar_route_rule(m, t):
    return m.P_sol[t] == m.S[t] * m.z_sol[t]
m.solar_route = Constraint(m.T, rule=solar_route_rule)

# (3) Power activation + limits + non-negativity already via domains
def grid_limit_rule(m, t):
    return m.P_grid[t] <= P_CH_MAX * m.z_grid[t]
m.grid_limit = Constraint(m.T, rule=grid_limit_rule)

def dis_limit_rule(m, t):
    return m.P_dis[t] <= P_DIS_MAX * m.z_dis[t]
m.dis_limit = Constraint(m.T, rule=dis_limit_rule)

# (4) Energy bounds
def energy_bounds_rule(m, t):
    return (E_MIN, m.E[t], E_MAX)
m.energy_bounds = Constraint(m.T, rule=energy_bounds_rule)

# (5) Initial condition
m.E_init = Constraint(expr=m.E[1] == E0)

# (6) Energy balance for t>=2
def energy_balance_rule(m, t):
    if t == 1:
        return Constraint.Skip
    P_ch = m.P_grid[t] + m.P_sol[t]
    return m.E[t] == (
        m.E[t-1]
        + ETA_CH * P_ch * DT
        - (m.P_dis[t] * DT) / ETA_DIS
        - P_AUX * DT
        - m.E[t-1] * R_SD * DT
    )
m.energy_balance = Constraint(m.T, rule=energy_balance_rule)

# -----------------------------
# CYCLE COUNTING (your "flag" method)
# -----------------------------

# Helper: c_t = z_sol + z_grid ; d_t = z_dis

# Set initial flag (safe default)
m.q_init = Constraint(expr=m.q[1] == 0)

# (a) Charging turns flag ON: q_t >= c_t
def q_on_rule(m, t):
    c_t = m.z_sol[t] + m.z_grid[t]
    return m.q[t] >= c_t
m.q_on = Constraint(m.T, rule=q_on_rule)

# (b) Discharging turns flag OFF: q_t <= 1 - d_t
def q_off_rule(m, t):
    d_t = m.z_dis[t]
    return m.q[t] <= 1 - d_t
m.q_off = Constraint(m.T, rule=q_off_rule)

# (c) Idle does not reset the flag: q_t >= q_{t-1} - d_t   (t>=2)
def q_hold_rule(m, t):
    if t == 1:
        return Constraint.Skip
    d_t = m.z_dis[t]
    return m.q[t] >= m.q[t-1] - d_t
m.q_hold = Constraint(m.T, rule=q_hold_rule)

# (d) Flag can only turn on if charging occurs: q_t <= q_{t-1} + c_t (t>=2)
def q_limit_rule(m, t):
    if t == 1:
        return Constraint.Skip
    c_t = m.z_sol[t] + m.z_grid[t]
    return m.q[t] <= m.q[t-1] + c_t
m.q_limit = Constraint(m.T, rule=q_limit_rule)

# Start of discharge segment: s_dis_t >= d_t - d_{t-1}, set s_dis_1 = 0
m.s_dis_init = Constraint(expr=m.s_dis[1] == 0)

def s_dis_rule(m, t):
    if t == 1:
        return Constraint.Skip
    return m.s_dis[t] >= m.z_dis[t] - m.z_dis[t-1]
m.s_dis_def = Constraint(m.T, rule=s_dis_rule)

# Cycle event: cycle_t = AND(s_dis_t, q_{t-1}) for t>=2
m.cycle_init = Constraint(expr=m.cycle[1] == 0)

def cycle_and_1(m, t):
    if t == 1:
        return Constraint.Skip
    return m.cycle[t] <= m.s_dis[t]
def cycle_and_2(m, t):
    if t == 1:
        return Constraint.Skip
    return m.cycle[t] <= m.q[t-1]
def cycle_and_3(m, t):
    if t == 1:
        return Constraint.Skip
    return m.cycle[t] >= m.s_dis[t] + m.q[t-1] - 1

m.cycle_and1 = Constraint(m.T, rule=cycle_and_1)
m.cycle_and2 = Constraint(m.T, rule=cycle_and_2)
m.cycle_and3 = Constraint(m.T, rule=cycle_and_3)

# Daily limit (single day in this run)
m.daily_cycles = Constraint(expr=sum(m.cycle[t] for t in m.T) <= MAX_CYCLES_PER_DAY)

# -----------------------------
# OBJECTIVE
# -----------------------------
# Max sum p_t*(P_dis - P_grid). (DT omitted; constant scaling)
m.obj = Objective(expr=sum(m.p[t] * (m.P_dis[t] - m.P_grid[t]) for t in m.T), sense=maximize)

# -----------------------------
# SOLVE
# -----------------------------
solver = SolverFactory("highs")
res = solver.solve(m, tee=True)

# -----------------------------
# EXPORT RESULTS
# -----------------------------
out = df.copy()
out["P_grid_MW"] = [value(m.P_grid[t]) for t in m.T]
out["P_dis_MW"]  = [value(m.P_dis[t]) for t in m.T]
out["z_sol"]     = [int(round(value(m.z_sol[t]))) for t in m.T]
out["z_grid"]    = [int(round(value(m.z_grid[t]))) for t in m.T]
out["z_dis"]     = [int(round(value(m.z_dis[t]))) for t in m.T]
out["E_MWh"]     = [value(m.E[t]) for t in m.T]
out["q_flag"]    = [int(round(value(m.q[t]))) for t in m.T]
out["s_dis"]     = [int(round(value(m.s_dis[t]))) for t in m.T]
out["cycle"]     = [int(round(value(m.cycle[t]))) for t in m.T]

# Profit per step (units scaled by DT if you want currency/MWh consistency)
out["profit_step"] = out["price_intraday"] * (out["P_dis_MW"] - out["P_grid_MW"]) * DT

out.to_csv(OUTPUT_CSV, index=False)

total_profit = out["profit_step"].sum()
total_cycles = out["cycle"].sum()

print(f"Saved: {OUTPUT_CSV}")
print(f"Total profit (with DT scaling): {total_profit:.2f}")
print(f"Total cycles counted: {int(total_cycles)}")
