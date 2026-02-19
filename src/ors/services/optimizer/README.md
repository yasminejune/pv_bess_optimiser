# BESS Intraday Optimizer
## Δt = 15 minutes · 6h Energy Capacity · Max 3 Cycles per Day
## Solar can be sold directly (with losses) · No curtailment
## Battery charging cannot mix Grid + Solar in the same interval

---

# 1) Time Structure
- Time steps: t = 1, …, T  
- Time resolution: 15 minutes  
- Δt = 0.25 hours  
- T = 96 for one day  

---

# 2) Inputs (Given Data)
- p_t : intraday electricity price at time t  
- S_t ≥ 0 (MW): available solar power at time t  

---

# 3) Fixed Parameters

## Battery Power
- Rated Power = 100 MW  
- Max charge power = P_ch_max = 100 MW  
- Max discharge power = P_dis_max = 100 MW  

## Efficiencies
- Charge efficiency: η_ch = 0.97  
- Discharge efficiency: η_dis = 0.97  
- Solar direct sale efficiency: η_sol_sell = 0.97  

## Losses
- Auxiliary load = 0.5% of rated power  
  → P_aux = 0.005 × 100 = 0.5 MW  

- Self-discharge rate = r_sd = 0.0005 per hour  

## Energy Capacity (6h duration)
- E_cap = 100 MW × 6 h = 600 MWh  
- E_min = 60 MWh (10%)  
- E_max = 540 MWh (90%)  
- Initial energy E_0 = 300 MWh (50%)  

---

# 4) Variables

## Power variables
- P_grid_t ≥ 0 (MW)  
  Grid power used to charge the battery (only if in grid-charging mode)

- P_dis_t ≥ 0 (MW)  
  Battery discharge power sold to the grid (only if in discharge mode)

- P_sol→bat_t ≥ 0 (MW)  
  Solar power sent into the battery (only if in solar-to-battery charging mode)

- P_sol→sell_t ≥ 0 (MW)  
  Solar power sold directly to grid (allowed in any battery mode)

## Battery mode binaries (mutually exclusive)
- z_grid_t ∈ {0,1}  → battery charges from grid  
- z_solbat_t ∈ {0,1} → battery charges from solar  
- z_dis_t ∈ {0,1}   → battery discharges to grid  

Battery is idle when all three are 0.

## State variable
- E_t (MWh) → battery stored energy at end of interval t  

---

# 5) Operating Constraints

## 5.1 Battery mode exclusivity
Battery can do at most one of: grid-charge, solar-charge, discharge, or idle:

z_grid_t + z_solbat_t + z_dis_t ≤ 1   ∀ t

This enforces the rule: **no simultaneous grid + solar charging**, and no charge while discharging.

---

## 5.2 Solar balance (no curtailment allowed)
All solar must be allocated:

P_sol→bat_t + P_sol→sell_t = S_t   ∀ t

Meaning:
- If S_t is produced, it must either go into the battery or be sold directly.

---

## 5.3 Power activation and limits (non-negativity included)

### Grid charging power
0 ≤ P_grid_t ≤ P_ch_max · z_grid_t = 100 · z_grid_t

### Solar-to-battery charging power
0 ≤ P_sol→bat_t ≤ P_ch_max · z_solbat_t = 100 · z_solbat_t

### Battery discharge power
0 ≤ P_dis_t ≤ P_dis_max · z_dis_t = 100 · z_dis_t

### Total battery charging power (for the energy balance)
P_ch_t = P_grid_t + P_sol→bat_t

Because of exclusivity, at most one of {P_grid_t, P_sol→bat_t} can be positive in a given interval.

---

# 6) Battery Energy Balance (Core Physics)

For t ≥ 2:

E_t =
E_{t−1}
+ η_ch · P_ch_t · Δt
− (P_dis_t · Δt) / η_dis
− P_aux · Δt
− E_{t−1} · r_sd · Δt

with Δt = 0.25 hours.

---

## Explanation of Each Term

### 1) Previous Energy
E_{t−1}  
Energy stored at end of previous interval.

### 2) Charging Energy Added
η_ch · P_ch_t · Δt  
- P_ch_t is charging power (MW)  
- Δt converts MW to MWh  
- η_ch accounts for charging losses  
Only a fraction of incoming energy is stored.

### 3) Discharge Energy Removed
(P_dis_t · Δt) / η_dis  
To sell P_dis_t·Δt MWh to the grid, the battery must remove more internal energy due to losses.

### 4) Auxiliary Consumption
P_aux · Δt  
Fixed parasitic consumption (HVAC, controls), always present.

### 5) Self-Discharge
E_{t−1} · r_sd · Δt  
Proportional decay of stored energy over time.

---

## Initial Condition
E_1 = E_0 = 300

## Energy Bounds
60 ≤ E_t ≤ 540   ∀ t

---

# 7) Terminal Constraint (Prevent End-of-Day Dump)

E_T ≥ 0.4 · E_cap

Since E_cap = 600 MWh:
E_T ≥ 240 MWh

---

# 8) Objective Function

Maximize daily profit:

max Σ_t p_t · ( P_dis_t + η_sol_sell · P_sol→sell_t − P_grid_t )

---

## Objective Explanation
- Battery discharge revenue: p_t · P_dis_t  
- Solar direct sale revenue (net of losses): p_t · η_sol_sell · P_sol→sell_t  
- Grid purchase cost: p_t · P_grid_t  
- Solar sent to battery does not generate immediate revenue; its value is realized if later discharged.

---

# 9) Cycle Counting Logic (Max 3 Cycles per Day)

We count a cycle as: **a discharge segment that occurs after the battery has charged at least once since the last discharge**.
Idle time between charge and discharge does not create a new cycle.

## 9.1 Aggregate charge/discharge modes
Charging indicator (battery is charging from either source):
c_t = z_grid_t + z_solbat_t

Discharging indicator:
d_t = z_dis_t

Note: by exclusivity, c_t + d_t ≤ 1.

---

## 9.2 Charge-flag variable
q_t ∈ {0,1}

Meaning:
- q_t = 1 → battery has charged since the last discharge  
- q_t = 0 → no charge pending

Constraints:

Charging turns flag ON:
q_t ≥ c_t

Discharging turns flag OFF:
q_t ≤ 1 − d_t

Idle preserves flag:
q_t ≥ q_{t−1} − d_t    ∀ t ≥ 2

Flag consistency:
q_t ≤ q_{t−1} + c_t    ∀ t ≥ 2

---

## 9.3 Detect start of discharge
s_dis_t ∈ {0,1}

s_dis_t ≥ d_t − d_{t−1}    ∀ t ≥ 2

---

## 9.4 Define a cycle event
cycle_t ∈ {0,1}

cycle_t ≤ s_dis_t  
cycle_t ≤ q_{t−1}  
cycle_t ≥ s_dis_t + q_{t−1} − 1    ∀ t ≥ 2  

---

## 9.5 Rolling daily cycle limit

Rather than summing cycles over a fixed calendar day, we enforce a **rolling 24-hour window** at every time step. This ensures the limit holds even when the optimization horizon spans multiple days (e.g. 48h).

N_day = 24 / Δt = 96 (number of periods in 24 hours).

For each t:

Σ_{k = max(1, t − N_day + 1)}^{t} cycle_k ≤ 3

At early time steps (t < N_day), the window simply starts at k = 1.
For t ≥ N_day, the window covers exactly the most recent 96 periods (24 hours).

---

# 10) Modeling Assumptions
- Solar power S_t ≤ 100 MW
- Solar cannot be curtailed (must be allocated to battery or direct sale)
- Solar direct sale allowed regardless of battery mode
- No ramp-rate limits
- No degradation cost beyond the cycle limit
- No thermal derating or forced outages

---

## Platform prerequisites and installation notes

This project is intended to run on Windows, macOS, and Linux. A few external/tooling prerequisites are platform-specific — follow the steps below for a smooth install.


If you prefer pip, prefer pip installing wheels in a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\Activate     # Windows (PowerShell: .\.venv\Scripts\Activate.ps1)

pip install --upgrade pip
pip install -e .[dev,ml]    # avoid `solvers` unless you installed system solvers (see below)
```

Solver notes (important) - Read the following only if the above is insufficient in satisfying all dependencies.
- GLPK (default solver in `pyproject.toml`): the pyomo `glpk` backend expects the GLPK executable (`glpsol`) on PATH.
  - macOS (Homebrew): `brew install glpk`
  - Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y glpk-utils libglpk-dev`
  - Windows: install via conda (`conda install -c conda-forge glpk`) or download a prebuilt binary and add it to PATH. Using WSL is an alternate route.
- HiGHS (`highspy`): available as wheels on many platforms; prefer conda-forge if pip wheel is unavailable.
- NEOS solver manager: the project defaults to `neos` in `pyproject.toml`. NEOS requires network access; if you are offline or behind a firewall, change `solver_manager` to `local` and configure a local solver.

Troubleshooting tips
- If pip build fails for `numpy`/`pandas`/`highspy`, install prebuilt packages via conda or use a Python version with available wheels (e.g., 3.10–3.12).
- To avoid the `glpk` executable requirement, change the default solver in your runtime (or in `pyproject.toml`) to a solver you have installed locally.

Packaging note
- The `all` extra previously referenced an undefined `ors` extra; this has been removed. Use `pip install -e .[all]` after installing any required system solvers, or install extras selectively (e.g., `pip install -e .[dev,ml]`).
