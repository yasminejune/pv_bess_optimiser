# BESS Optimizer Backtesting

Rolling-horizon backtest for the Battery Energy Storage System (BESS) optimizer.
Simulates realistic battery dispatch over a historical window and measures profit.

---

## Overview

The backtest replays historical electricity prices over a chosen time window and
runs the MILP optimizer repeatedly, committing a limited number of actions before
re-planning. This mirrors how the live system works: the optimizer sees a 24-hour
forecast horizon but only commits the next block of actions before receiving new
information and re-planning.

Two **price source** modes are supported:

| Mode | Optimizer sees | Profit settled at |
|---|---|---|
| `perfect` | True historical prices (upper bound) | True prices |
| `predicted` | LGBM model forecast (`use_csv=True`) | True prices |

Running both with the same `--seed` on the same window isolates the **cost of
forecast error** — the difference in profit is the value of perfect information.

---

## Prerequisites

From the repo root, install dependencies:

```bash
pip install -r requirements.txt
```

A linear solver must be available. The script tries each in order:
`highs` (via `highspy`), `glpk`, `cbc`, `gurobi`, `cplex`.

```bash
pip install highspy   # recommended
```

---

## Usage

Run from the **repo root**:

```bash
# Perfect-information baseline — 1 week, re-plan every 15 min
python tests/backtesting/backtest.py --mode 1week --seed 42

# Predicted-price run — same window, optimizer uses LGBM forecast
python tests/backtesting/backtest.py --mode 1week --seed 42 --price-source predicted

# 4-month run, re-plan every 4 hours
python tests/backtesting/backtest.py --mode 4month --seed 42

# With per-step debug trace
python tests/backtesting/backtest.py --mode 1week --seed 42 --trace

# Custom output path
python tests/backtesting/backtest.py --mode 1week --seed 42 --output results/my_run.csv
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--mode` | *(required)* | `4month` or `1week` (see below) |
| `--seed` | `None` (random) | Integer seed for window selection — use the same seed on both price-source runs to get the same window |
| `--price-source` | `perfect` | `perfect` or `predicted` |
| `--price-csv` | `Data/price_data_rotated_2d.csv` | Path to historical price CSV |
| `--output` | auto | Path for per-step detail CSV |
| `--solar-chunk-days` | `1` | Days per PV API request chunk |
| `--trace` | off | Write per-step optimizer debug trace |
| `--trace-output` | auto | Path for trace CSV (only used with `--trace`) |

---

## Simulation modes

### `4month` — 120-day run
- Picks a random 120-day window from the historical price data.
- The optimizer re-plans every **4 hours** (at 00:00, 04:00, 08:00, 12:00,
  16:00, 20:00 UTC). The next 16 × 15-min periods are committed; the remaining
  plan is discarded.
- Approximately 720 optimizer calls total.

### `1week` — 7-day run
- Picks a random 7-day window.
- The optimizer re-plans every **15 minutes** — only the very next action is
  committed before re-planning.
- 672 optimizer calls total.

---

## Price sources

### `perfect` (default)
The true historical prices from `price_data_rotated_2d.csv` are fed directly
to the optimizer. This is the **perfect-information baseline** — an upper bound
on achievable profit because the optimizer sees the future exactly.

### `predicted`
At each re-plan step, the LGBM recursive price prediction model is called with
`reference_time` set to the current simulation timestamp and `use_csv=True`.
This means features are built from the static training CSVs in `Data/`,
reproducing the model's predictions for that historical period without making
live API calls.

The optimizer dispatches based on these forecasts, but **profit is always
settled at the true historical price**. Forecast errors therefore translate
directly into sub-optimal charge/discharge timing and reduced profit.

---

## Solar generation

PV output is pre-generated once for the entire simulation window by calling
`generate_pv_power_for_date_range` in 1-day chunks before the backtest loop
begins. This avoids repeated API calls and mirrors the live optimizer's
`load_inputs` path (`kW → MW` conversion applied).

If PV generation fails for any chunk, that chunk defaults to 0 MW solar and
the backtest continues.

---

## Output files

When `--output` is not specified, files are written to `tests/backtesting/`
with auto-generated names that encode the mode, price source, and seed:

```
backtest_detail_<mode>_<price_source>_<seed>.csv   # per-step detail
backtest_summary_<mode>_<price_source>_<seed>.csv  # daily summary
backtest_detail_<mode>_<price_source>_<seed>_trace.csv  # debug trace (--trace only)
```

### Detail CSV columns

| Column | Description |
|---|---|
| `timestamp` | UTC timestamp of the 15-min period |
| `price` | True settlement price (£/MWh) |
| `P_grid_MW` | Grid → battery charging power (MW) |
| `P_dis_MW` | Battery → grid discharge power (MW) |
| `P_sol_bat_MW` | Solar → battery charging power (MW) |
| `P_sol_sell_MW` | Solar → grid direct sell power (MW) |
| `z_grid` / `z_solbat` / `z_dis` | Binary mode flags |
| `cycle` | 1 if a charge-discharge cycle completed this step |
| `E_MWh` | Battery state of charge after the step (MWh) |
| `profit_step` | Revenue for this step (£), settled at true price |

### Summary CSV columns

| Column | Description |
|---|---|
| `Date` | Calendar date |
| `profit_GBP` | Total profit for the day (£) |
| `n_cycles` | Charge-discharge cycles completed |
| `avg_price_GBP_MWh` | Average settlement price (£/MWh) |
| `min_SOC_MWh` / `max_SOC_MWh` | Daily SOC range (MWh) |

---

## Comparing perfect vs predicted

Run both modes with the same seed to compare on identical windows:

```bash
python tests/backtesting/backtest.py --mode 1week --seed 42 --price-source perfect
python tests/backtesting/backtest.py --mode 1week --seed 42 --price-source predicted
```

The summary CSVs will be named `..._perfect_42.csv` and `..._predicted_42.csv`.
The difference in `profit_GBP` totals is the **value of perfect information**
(VPI) for that window — the maximum benefit that could be gained from a
perfect price forecast.

---

## Key model parameters

These are defined in `src/ors/services/optimizer/optimizer.py` and mirrored
as constants in `backtest.py`:

| Parameter | Value | Description |
|---|---|---|
| `E_CAP` | 600 MWh | Total battery capacity |
| `E_MIN` / `E_MAX` | 60 / 540 MWh | Operational SOC bounds |
| `P_CH_MAX` / `P_DIS_MAX` | 100 MW | Charge / discharge power limit |
| `ETA_CH` / `ETA_DIS` | 0.97 | Round-trip efficiency |
| `MAX_CYCLES_PER_DAY` | 3 | Cycle limit per 23:00–23:00 day |
| `DT` | 0.25 h | Time step (15 min) |
| `N_D` | 96 | Periods per day |
| `H_30` | 4.8 h | Window for `p_30` terminal-value averaging |
