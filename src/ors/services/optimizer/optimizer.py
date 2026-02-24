import pandas as pd
from datetime import datetime
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

# Import battery integration functions
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'battery_to_optimization'))
from battery_inference import (
    load_optimizer_battery_config,
    create_enhanced_optimizer_output,
    validate_optimizer_energy_balance
)

# -----------------------------
# CONFIG
# -----------------------------
INPUT_CSV = "tests/data/optimizer/bess_test_data_intraday_15min.csv"  # columns: timestamp, price_intraday, solar_MW
OUTPUT_CSV = "tests/data/optimizer/bess_solution_v2_15min_1day.csv"
BATTERY_STEP_CSV = "src/ors/services/battery_to_optimization/battery_storage.csv"

# Load battery configuration from battery module
print("Loading battery configuration...")
try:
    battery_params, sim_defaults = load_optimizer_battery_config()
    print(f"✓ Loaded battery config: {battery_params.p_rated_mw}MW / {battery_params.e_cap_mwh}MWh")
except Exception as e:
    print(f"⚠ Could not load battery config: {e}")
    print("Using default parameters...")
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
E0 = E_CAP * 0.5  # Start at 50% SOC
E_TERMINAL_MIN = E_CAP * 0.4  # 40% minimum terminal SOC

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
print("\n🔄 Solving optimization model...")

# Try multiple solvers in order of preference
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
    print("❌ No suitable solver found. Please install one of:")
    print("  - HiGHS: pip install highspy") 
    print("  - GLPK: conda install glpk or brew install glpk")
    print("  - CBC: conda install coincbc")
    raise RuntimeError("No optimization solver available")

res = solver.solve(m, tee=True)  # tee=True shows progress

if res.solver.termination_condition.value == "optimal":
    print("✓ Optimization completed successfully")
else:
    print(f"⚠ Solver status: {res.solver.termination_condition}")

# -----------------------------
# EXPORT RESULTS
# -----------------------------
print("\n📊 Processing results...")
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

# Save original format
out.to_csv(OUTPUT_CSV, index=False)
print(f"✓ Saved original format: {OUTPUT_CSV}")

# Create enhanced output with battery module integration
print("\n🔋 Creating enhanced battery module output...")
try:
    # Create timestamp for the simulation start (example: start of day)
    start_datetime = datetime(2024, 6, 15, 0, 0, 0)
    
    # Export using battery module functions with step-by-step logging to battery_to_optimization folder
    export_results = create_enhanced_optimizer_output(
        df_results=out,
        csv_path=BATTERY_STEP_CSV,
        params=battery_params,
        dt_hours=DT,
        start_datetime=start_datetime,
        validate=True
    )
    
    print(f"✓ Battery logs saved: {BATTERY_STEP_CSV}")
    print(f"  Steps processed: {export_results['num_steps']}")
    print(f"  Battery config: {export_results['params_used']}")
    
    # Display validation results
    if 'validation' in export_results:
        validation = export_results['validation']
        if validation['is_valid']:
            print(f"✓ Energy balance validation: PASSED")
            print(f"  Max error: {validation['max_error']:.6f} MWh")
        else:
            print(f"⚠ Energy balance validation: FAILED")
            print(f"  Failed steps: {validation['summary']['failed_steps']}")
            print(f"  Max error: {validation['max_error']:.6f} MWh")
            for error in validation['errors'][:3]:  # Show first 3 errors
                print(f"    Step {error['step']}: {error['error']:.6f} MWh difference")
    
except Exception as e:
    print(f"⚠ Enhanced output failed: {e}")
    print("Original CSV output is still available.")

# Summary
print(f"\n📈 Optimization Summary:")
print(f"Total profit (Δt-scaled): €{out['profit_step'].sum():.2f}")
print(f"Total cycles counted: {int(out['cycle'].sum())}")
print(f"Terminal energy E_T: {out['E_MWh'].iloc[-1]:.2f} MWh (min {E_TERMINAL_MIN:.2f})")
print(f"Energy utilization: {((out['E_MWh'].max() - out['E_MWh'].min()) / E_CAP * 100):.1f}% of capacity")

# Print a few sample timesteps to show the step-by-step behavior
print(f"\n📊 Sample Timesteps:")
print(f"{'Time':<8} {'Price':<8} {'Solar':<8} {'Grid':<8} {'Discharge':<10} {'Energy':<8} {'Mode':<12}")
print("-" * 70)
for i in range(0, min(12, len(out)), 3):  # Show every 3rd timestep for first 12
    row = out.iloc[i]
    hour = int(i * DT)
    minute = int((i * DT % 1) * 60)
    mode = "Idle"
    if row['z_grid'] > 0.5: mode = "Grid→Batt"
    elif row['z_solbat'] > 0.5: mode = "Solar→Batt"
    elif row['z_dis'] > 0.5: mode = "Batt→Grid"    
    
    print(f"{hour:02d}:{minute:02d}    {row['price_intraday']:6.2f}  {row['solar_MW']:6.1f}  {row['P_grid_MW']:6.1f}  {row['P_dis_MW']:8.1f}  {row['E_MWh']:6.1f}  {mode:<12}")

print(f"\n✅ Optimization completed successfully!")
print(f"📄 Files created:")
print(f"  - {OUTPUT_CSV} (original format)")
print(f"  - {BATTERY_STEP_CSV} (detailed battery storage logs)")
