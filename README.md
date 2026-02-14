# BESS Intraday Optimizer
## Δt = 15 minutes · 6h Energy Capacity · Max 3 Cycles per Day

---

## 0) Time structure
- Time steps: t = 1, …, T  
- Time resolution: 15 minutes  
- Δt = 0.25 hours  
- Days are defined as subsets of time steps (used for daily cycle limits)

---

## 1) Inputs (given data)
- p_t : intraday spot price at time t  
- S_t ≥ 0 (MW): available solar power at time t  

---

## 2) Fixed parameters (prototype)

### Power & efficiency
- Rated power: 100 MW  
- Max charge power: P_ch_max = 100 MW  
- Max discharge power: P_dis_max = 100 MW  
- Charge efficiency: η_ch = 0.97  
- Discharge efficiency: η_dis = 0.97  

### Auxiliary losses
- Aux load fraction: a = 0.5 % of rated power  
- P_aux = a · P_rated = 0.005 · 100 = 0.5 MW  

### Self-discharge
- r_sd = 0.0005 per hour (0.05 % / h)

### Energy capacity (6h duration)
- E_cap = 100 MW × 6 h = 600 MWh  
- E_min = 10 % · E_cap = 60 MWh  
- E_max = 90 % · E_cap = 540 MWh  
- Initial energy: E_0 = 300 MWh  

---

## 3) Decision variables (controls)

### Power decisions
- P_grid_t ≥ 0 (MW): power purchased from grid to charge battery  
- P_dis_t ≥ 0 (MW): power discharged and sold to grid  

### Mode-selection binaries
- z_sol_t ∈ {0,1}: solar → battery  
- z_grid_t ∈ {0,1}: grid → battery  
- z_dis_t ∈ {0,1}: battery → grid  

Idle occurs when all three binaries are zero.

---

## 4) State variables (not freely chosen)
- E_t (MWh): energy stored in the battery at end of interval t  

---

## 5) Operating logic

### 5.1 Mode exclusivity
Only one action can occur per interval:

z_sol_t + z_grid_t + z_dis_t ≤ 1   ∀ t

---

### 5.2 Solar routing
If solar mode is active, all available solar enters the battery:

P_sol_t = S_t · z_sol_t

---

### 5.3 Power activation and limits

Grid charging:
0 ≤ P_grid_t ≤ 100 · z_grid_t

Discharging:
0 ≤ P_dis_t ≤ 100 · z_dis_t

Total charging power:
P_ch_t = P_grid_t + P_sol_t

---

## 6) Energy balance (battery dynamics)

E_t =
E_{t-1}
+ η_ch · P_ch_t · Δt
− (P_dis_t · Δt) / η_dis
− P_aux · Δt
− E_{t-1} · r_sd · Δt

with Δt = 0.25 hours.

### Initial condition
E_1 = E_0 = 300

### Energy bounds
60 ≤ E_t ≤ 540   ∀ t

---

## 7) Objective function (intraday arbitrage)

Maximize profit from selling energy and paying for grid purchases:

max Σ_t p_t · (P_dis_t − P_grid_t)

---

## 8) Cycle-count logic (max 3 cycles per day)

### 8.1 Aggregate charge / discharge modes
Charging (solar or grid):
c_t = z_sol_t + z_grid_t

Discharging:
d_t = z_dis_t

Constraint:
c_t + d_t ≤ 1   ∀ t

---

### 8.2 Charge-flag variable
q_t ∈ {0,1}

Interpretation:
- q_t = 1 → there has been charging since the last discharge  
- q_t = 0 → no charge pending

Constraints:

Charging turns flag ON:
q_t ≥ c_t

Discharging turns flag OFF:
q_t ≤ 1 − d_t

Idle does not reset the flag:
q_t ≥ q_{t−1} − d_t    ∀ t ≥ 2

Flag can only turn on if charging occurs:
q_t ≤ q_{t−1} + c_t    ∀ t ≥ 2

---

### 8.3 Detect start of discharge
s_dis_t ∈ {0,1}

s_dis_t ≥ d_t − d_{t−1}    ∀ t ≥ 2

---

### 8.4 Cycle definition
cycle_t ∈ {0,1}

cycle_t ≤ s_dis_t
cycle_t ≤ q_{t−1}
cycle_t ≥ s_dis_t + q_{t−1} − 1    ∀ t ≥ 2

Interpretation:
A cycle is counted only when:
- a discharge starts, and
- there was charging since the previous discharge
Idle time between charge and discharge does not create a new cycle.

---

### 8.5 Daily cycle limit
For each day D:

Σ_{t ∈ D} cycle_t ≤ 3

---

## 9) Explicit assumptions
- Solar power S_t ≤ 100 MW  
- Solar cannot be exported directly  
- No ramp-rate limits  
- No degradation cost (only cycle count)  
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
