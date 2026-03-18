# Optimizer Service — Change Log

Branch: `regressor_pred`

This document covers all changes made to the optimizer service to wire in the
LightGBM price prediction model and fix runtime errors that prevented the full
pipeline from running.

---

## Files changed

### `integration.py` — **LGBM model wired as default**

**Before:** `create_input_df` called `run_inference(**kwargs)` with no
`model_path`, relying on silent auto-detection inside `run_inference`.  If the
LGBM model directory was missing or empty, the pipeline would silently fall
back to the legacy XGBoost model with no error or warning.  A `use_historic`
flag was computed but both branches of the `if/else` made the identical
`run_inference` call — dead code.

**After:**

- Added `from pathlib import Path` and
  `from ors.services.price_inference.live_inference import LGBM_MODEL_DIR`.
- Added `model_path: Path = LGBM_MODEL_DIR` as an explicit keyword parameter.
  LGBM is now the declared default, not a silent fallback.
- `run_inference` receives `model_path=model_path` explicitly.
- Removed the `use_historic` flag and its identical `if/else` branches.
- Updated docstring to document the new `model_path` parameter with usage
  examples for LGBM directory, specific `.joblib` file, and legacy XGBoost.

**Signature change:**

```python
# Before
def create_input_df(config, *, client, start_datetime, end_datetime, **kwargs)

# After
def create_input_df(config, *, client, start_datetime, end_datetime,
                    model_path: Path = LGBM_MODEL_DIR, **kwargs)
```

---

### `optimizer.py` — **timezone bug fixes + no double inference**

#### Bug 1 — `TypeError`: tz-naive vs tz-aware comparison (`optimizer.py:158`)

**Error:**
```
TypeError: Invalid comparison between dtype=datetime64[ns, UTC] and datetime
```

**Root cause:** `load_inputs` built `day_start` from `datetime.now()` which is
tz-naive.  The output CSV timestamp column is tz-aware UTC (written by
`create_input_df`), so the pandas `>=` comparison raised a `TypeError`.

**Fix (three changes):**

1. Added `timezone` to the `datetime` import:
   ```python
   # Before
   from datetime import datetime, timedelta
   # After
   from datetime import datetime, timedelta, timezone
   ```

2. Changed `now` to be tz-aware by default:
   ```python
   # Before
   now = now_dt or datetime.now()
   # After
   now = now_dt or datetime.now(timezone.utc)
   ```
   Because `day_start` is derived from `now` via `.replace(...)`, which
   preserves the timezone, `day_start` is now also tz-aware UTC.

3. Parsed the output CSV timestamps explicitly as UTC:
   ```python
   # Before
   df_output["timestamp"] = pd.to_datetime(df_output["timestamp"])
   # After
   df_output["timestamp"] = pd.to_datetime(df_output["timestamp"], utc=True)
   ```

#### Bug 2 — `SettingWithCopyWarning` (`optimizer.py:114`)

**Warning:**
```
SettingWithCopyWarning: A value is trying to be set on a copy of a slice from
a DataFrame.
```

**Root cause:** `create_input_df` returns `df[:96]` — a slice, not a copy.
Assigning `df["timestamp"] = pd.to_datetime(...)` into the slice triggered the
pandas warning.

**Fix:** Call `.copy()` immediately after `create_input_df`:
```python
# Before
df = create_input_df(**kwargs)
# After
df = create_input_df(**kwargs).copy()
```

#### Bug 3 — Inference ran twice per optimizer invocation

**Symptom:** The full LGBM inference pipeline (API fetches + model load +
prediction) executed twice on each run — once inside `load_inputs` and once
again in `__main__` after the solver finished, just to recover timestamps for
the output CSV.

**Root cause:** `__main__` had a bare `df = create_input_df()` call after
`solver.solve()`:

```python
# After solver
df = create_input_df()          # ← second full inference run
df["timestamp"] = pd.to_datetime(df["timestamp"])
out = df.copy()
...
```

**Fix:** Return `df` as a sixth value from `load_inputs` and reuse it:

```python
# load_inputs return type
# Before:  tuple[dict, dict, float, int, int]
# After:   tuple[dict, dict, float, int, int, pd.DataFrame]

return price, solar, p_30, cycles_used_today, t_boundary, df
```

```python
# __main__ call site
# Before:
price, solar, p_30, cycles_used_today, t_boundary = load_inputs(...)
...
df = create_input_df()     # second run

# After:
price, solar, p_30, cycles_used_today, t_boundary, df = load_inputs(...)
# df reused directly — no second inference call
```

---

## How to run

```bash
# From repo root — runs live LGBM inference + HiGHS solver
python -m ors.services.optimizer.optimizer
```

Prerequisites (if not already done):

```bash
# 1. Install a solver
pip install highspy

# 2. Train the LGBM model
python -m ors.services.prediction.train_script --model-name lgbm_recursive
```

Output is written to `tests/data/optimizer/bess_solution_v2_15min_1day.csv`.

---

## Summary of all fixes

| File | Change | Reason |
|---|---|---|
| `integration.py` | Added `model_path` param defaulting to `LGBM_MODEL_DIR` | Make LGBM the explicit default, not a silent fallback |
| `integration.py` | Removed `use_historic` dead code | Both branches were identical |
| `integration.py` | Pass `model_path` to `run_inference` | Required for LGBM to be used |
| `optimizer.py` | Added `timezone` import | Required for fix below |
| `optimizer.py` | `datetime.now(timezone.utc)` | Fix tz-naive vs tz-aware `TypeError` |
| `optimizer.py` | `pd.to_datetime(..., utc=True)` on output CSV | Ensure tz-aware UTC when reading back previous results |
| `optimizer.py` | `.copy()` on `create_input_df()` result | Fix `SettingWithCopyWarning` (slice assignment) |
| `optimizer.py` | Return `df` from `load_inputs` | Eliminate duplicate inference run |
| `optimizer.py` | Remove second `create_input_df()` call in `__main__` | Inference now runs exactly once |

---

## Branch: `backtesting`

### `optimizer.py` — **cycle counting bug fix at `t=1`**

#### Bug — `cycle[1]` always 0 / cycle cap not enforced at horizon start

**Symptom:** The `n_cycles` column in backtest summary CSVs was 0 for every
day when running in `1week` mode (`--mode 1week`).  In `4month` mode the first
period of every 4-hour commit window was also silently missed.

**Root cause:** Five constraint rules in `build_model` contained
`if t == 1: return Constraint.Skip` because they reference `q[t-1]` or
`z_dis[t-1]`, which don't exist in the model at `t=1`.  As a result:

- `s_dis[1]` was unconstrained → never forced to 1, so always 0.
- `cycle[1]` was unconstrained → always 0 (no objective incentive to be 1).
- The `cycles_cur_day` cap summed `cycle[1..t_boundary]` but `cycle[1]` was
  always 0, so the optimizer could schedule an extra cycle in the first period
  without it counting against the cap.

In `1week` mode, `commit_periods=1` means only `t=1` is ever committed per
optimizer call, so every single cycle went undetected and the daily cap was
never enforced.

**Fix:** Added two new parameters to `build_model`:

```python
def build_model(
    price, solar, p_30, cycles_used_today, t_boundary,
    q_init: int = 0,      # charge flag before period 1
    z_dis_init: int = 0,  # discharge mode active before period 1
) -> ConcreteModel:
```

The five rules now use these constants instead of skipping at `t=1`:

| Rule | Before | After |
|---|---|---|
| `q_hold_rule` | skip at t=1 | `q[1] >= q_init - z_dis[1]` |
| `q_limit_rule` | skip at t=1 | `q[1] <= q_init + c_1` |
| `s_dis_rule` | skip at t=1 | `s_dis[1] >= z_dis[1] - z_dis_init` |
| `cycle_and2` | skip at t=1 | `cycle[1] <= q_init` |
| `cycle_and3` | skip at t=1 | `cycle[1] >= s_dis[1] + q_init - 1` |

`cycle_and1` (`cycle[t] <= s_dis[t]`) required no change as it never
referenced `t-1`.

Both parameters default to `0`, preserving backward compatibility for callers
that do not supply them (the optimizer assumes no prior charging and no prior
discharge at the start of any fresh horizon).

---

### `tests/backtesting/backtest.py` — **wiring + simulation-based cycle counter**

Two changes were made to the backtest to complement the optimizer fix:

**1. Pass pre-horizon state to the optimizer.**  `run_single_optimize` now
accepts `q_init` and `z_dis_init` and forwards them to `build_model`.  The
backtest loop maintains `q_sim` (bool) and `prev_z_dis_sim` (int) across steps
and passes them at each re-plan call:

```python
m, solve_info = run_single_optimize(
    solver, price_dict, solar_dict, p_30,
    cycles_for_model, t_boundary, e_soc,
    q_init=int(q_sim), z_dis_init=prev_z_dis_sim,
)
```

**2. Simulation-based cycle counter as safety net.**  Even with the optimizer
fixed, the backtest now independently detects cycle events from the actual
applied `z_dis` / `z_grid` / `z_solbat` flags, using the same AND-logic as
the optimizer (`cycle = start_of_discharge AND charged_since_last_discharge`).
This ensures `n_cycles` and `cycles_since_23` are always correct regardless of
solver numerical rounding:

```python
s_dis_now = int(z_dis == 1 and prev_z_dis_sim == 0)
cyc = int(s_dis_now and q_sim)
if z_dis:          q_sim = False
elif z_grid or z_solbat: q_sim = True
prev_z_dis_sim = z_dis
```

---

## Summary of all fixes (backtesting branch)

| File | Change | Reason |
|---|---|---|
| `optimizer.py` | Added `q_init`, `z_dis_init` params to `build_model` | Anchor cycle logic at horizon start |
| `optimizer.py` | Removed `t==1` skip from `q_hold`, `q_limit`, `s_dis`, `cycle_and2/3` | `cycle[1]` can now correctly be 1 |
| `backtest.py` | `run_single_optimize` accepts and forwards `q_init`, `z_dis_init` | Pass real pre-horizon state to optimizer |
| `backtest.py` | Simulation-based `cyc` counter from applied flags | Correct `n_cycles` / `cycles_since_23` regardless of which `t` is committed |

