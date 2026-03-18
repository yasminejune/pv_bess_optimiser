#!/usr/bin/env python3
"""
Backtesting script for the BESS optimizer.

Modes
-----
4month
    Random 4-month window (~120 days).
    Optimizer re-runs every 4 hours (at each 4-hour boundary: 00:00, 04:00,
    08:00, 12:00, 16:00, 20:00).  Only the next 4-hour block of planned
    actions is committed; the rest is discarded.

1week
    Random 1-week window (7 days).
    Optimizer re-runs every 15 minutes.  Only the single next action is
    committed.

Price source (--price-source)
-----------------------------
perfect (default)
    The true prices from price_data_rotated_2d.csv are fed to the optimizer.
    This is the perfect-information baseline — an upper bound on achievable
    profit.

predicted
    The LGBM price prediction model (``use_csv=True``) produces a 24-h
    forecast at each re-plan step.  These predicted prices drive the
    optimizer's decisions.  Profit is still settled at the true prices, so
    any forecast error translates directly into sub-optimal dispatch.
    Compare against the ``perfect`` run (same seed) to measure the value
    of perfect information.

Solar generation comes from generate_pv_power_for_date_range (the same
source used by the live optimizer via create_input_df).  It is
pre-generated once for the entire simulation window before the loop
starts, then sliced per optimizer call.  This mirrors load_inputs:
    df["generation_kw"] = df["generation_kw"] / 1000   # kW → MW

Usage (from repo root)
----------------------
    python tests/backtesting/backtest.py --mode 4month [--seed 42]
    python tests/backtesting/backtest.py --mode 1week  [--seed 42]
    python tests/backtesting/backtest.py --mode 1week  --price-source predicted --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pyomo.environ import SolverFactory, value
from pyomo.opt import SolverStatus, TerminationCondition

# ---------------------------------------------------------------------------
# Repo root on Python path
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

import ors.services.optimizer.optimizer as opt_module  # noqa: E402

# ---------------------------------------------------------------------------
# Constants (derived from example_full_config.json battery spec)
# ---------------------------------------------------------------------------
DT: float = 0.25  # 15-min time step in hours
N_D: int = 96  # periods per day
E_CAP: float = 200.0  # MWh — energy_capacity_mwh
E_MIN: float = 20.0  # MWh — 10% of E_CAP (min_soc_percent)
E_MAX: float = 180.0  # MWh — 90% of E_CAP (max_soc_percent)
P_CH_MAX: float = 50.0  # MW — rated_power_mw
P_DIS_MAX: float = 50.0  # MW — rated_power_mw
ETA_CH: float = 0.97  # charge_efficiency
ETA_DIS: float = 0.97  # discharge_efficiency
ETA_SOL_SELL: float = 0.97
P_AUX: float = 1.0  # MW — auxiliary_power_mw
R_SD: float = 0.0005  # self_discharge_rate_per_hour
H_30: float = 0.5  # hours — first 30 min of day for p_30 calculation
MAX_CYCLES_PER_DAY: int = 3

# Battery spec object passed to build_model (base; current_energy_mwh overridden per call)
_BASE_BATTERY_SPEC = types.SimpleNamespace(
    rated_power_mw=P_CH_MAX,
    energy_capacity_mwh=E_CAP,
    min_soc_percent=E_MIN / E_CAP * 100.0,
    max_soc_percent=E_MAX / E_CAP * 100.0,
    charge_efficiency=ETA_CH,
    discharge_efficiency=ETA_DIS,
    auxiliary_power_mw=P_AUX,
    self_discharge_rate_per_hour=R_SD,
    max_cycles_per_day=MAX_CYCLES_PER_DAY,
    current_energy_mwh=None,
)

PERIODS_PER_4H = 16  # 4 hours at 15-min resolution
PERIODS_PER_15MIN = 1

PRICE_CSV = REPO_ROOT / "Data" / "price_data_rotated_2d.csv"
HIST_DAYS = 30  # days of historical price context required for p_30


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
def get_solver():
    for name in ["highs", "glpk", "cbc", "gurobi", "cplex"]:
        try:
            s = SolverFactory(name)
            if s.available():
                print(f"Solver: {name}")
                return s
        except Exception:
            continue
    raise RuntimeError(
        "No solver found. Install one of: highs (highspy), glpk, cbc, gurobi, cplex."
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_price_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# p_30 computation
# ---------------------------------------------------------------------------
def compute_p30(df_all: pd.DataFrame, run_time: datetime) -> float:
    """
    Average price in the first H_30 hours of each day over the 30 days
    preceding run_time.  Used as the terminal-value reference price.
    """
    end = run_time - timedelta(days=1)
    start = run_time - timedelta(days=HIST_DAYS + 1)
    mask = (df_all["timestamp"] >= start) & (df_all["timestamp"] < end)
    df_hist = df_all.loc[mask].copy()

    if df_hist.empty:
        return float(df_all["price"].mean())

    df_hist["time_h"] = df_hist["timestamp"].dt.hour + df_hist["timestamp"].dt.minute / 60.0
    prices = df_hist.loc[df_hist["time_h"] < H_30, "price"]
    return float(prices.mean()) if not prices.empty else float(df_hist["price"].mean())


# ---------------------------------------------------------------------------
# Pre-generate solar for the entire simulation window (called once in main)
# ---------------------------------------------------------------------------
def _fetch_solar_chunk(
    config,
    chunk_start: datetime,
    chunk_end: datetime,
) -> pd.DataFrame:
    """
    Fetch one contiguous PV chunk and return a normalised DataFrame with
    columns ['timestamp_utc', 'generation_MW'].
    """
    from ors.services.weather_to_pv import generate_pv_power_for_date_range

    df = generate_pv_power_for_date_range(
        config=config,
        start_datetime=chunk_start,
        end_datetime=chunk_end,
    )
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    # kW → MW: identical to load_inputs line:
    #   df["generation_kw"] = df["generation_kw"] / 1000
    df["generation_MW"] = df["generation_kw"] / 1000.0
    return df[["timestamp_utc", "generation_MW"]].copy()


def pregenerate_solar(
    start_dt: datetime,
    end_dt: datetime,
    chunk_days: int = 1,
) -> pd.DataFrame | None:
    """
    Generate site-level PV output for [start_dt, end_dt + 1 day) by calling
    generate_pv_power_for_date_range in chunks of *chunk_days* days.

    Chunking is required because the PV weather model may have a limited
    forecast horizon; a single multi-month call would either fail or return
    unrealistic values for distant dates.

    Parameters
    ----------
    start_dt : first timestamp of the simulation window (UTC-aware).
    end_dt   : last day of the simulation window (UTC-aware).
               An extra day is appended so the final optimizer run still has
               a full 24-h look-ahead.
    chunk_days : size of each generation request in days (default 1).

    Returns
    -------
    DataFrame with columns ['timestamp_utc', 'generation_MW'], sorted and
    deduplicated, or None if every chunk fails.
    """
    from ors.config.pv_config import SiteType, get_pv_config

    try:
        config = get_pv_config(SiteType.BURST_1)
    except Exception as exc:
        print(f"  WARNING: Could not load PV config ({exc}). Solar will be 0.")
        return None

    # Build list of (chunk_start, chunk_end) pairs
    window_end = end_dt + timedelta(days=1)  # extra day for look-ahead
    boundaries: list[tuple[datetime, datetime]] = []
    cursor = start_dt
    while cursor < window_end:
        chunk_end = min(cursor + timedelta(days=chunk_days), window_end)
        boundaries.append((cursor, chunk_end))
        cursor = chunk_end

    total = len(boundaries)
    print(f"  Fetching PV in {total} chunk(s) of {chunk_days} day(s) each ...")

    collected: list[pd.DataFrame] = []
    failed = 0
    for idx, (cs, ce) in enumerate(boundaries, start=1):
        try:
            df_chunk = _fetch_solar_chunk(config, cs, ce)
            collected.append(df_chunk)
            print(f"    [{idx:>4d}/{total}] {cs.date()} -> {ce.date()}  ({len(df_chunk)} rows)")
        except Exception as exc:
            print(f"    [{idx:>4d}/{total}] {cs.date()} -> {ce.date()}  FAILED: {exc}")
            failed += 1

    if not collected:
        print("  WARNING: All PV chunks failed. Solar will be 0 for all periods.")
        return None

    if failed:
        print(f"  WARNING: {failed}/{total} chunk(s) failed; missing periods will use solar = 0.")

    df_pv = (
        pd.concat(collected, ignore_index=True)
        .drop_duplicates(subset="timestamp_utc")
        .sort_values("timestamp_utc")
        .reset_index(drop=True)
    )
    print(
        f"  Solar pre-generated: {len(df_pv)} rows total  "
        f"({df_pv['timestamp_utc'].min().date()} -> {df_pv['timestamp_utc'].max().date()})"
    )
    return df_pv


# ---------------------------------------------------------------------------
# Build 96-period price / solar dicts for the optimizer
# ---------------------------------------------------------------------------
def build_price_solar_dicts(
    df_all: pd.DataFrame,
    start_idx: int,
    df_solar: pd.DataFrame | None,
) -> tuple[dict[int, float], dict[int, float]]:
    """
    Extract N_D (96) price rows from df_all starting at start_idx.
    Solar values (MW) are sliced from df_solar by timestamp alignment.
    If df_solar is None or alignment fails, solar defaults to 0.
    Price gaps at the tail are padded with the last known price.
    """
    rows = df_all.iloc[start_idx : start_idx + N_D].reset_index(drop=True)
    last_price = float(rows["price"].iloc[-1]) if not rows.empty else 100.0

    price: dict[int, float] = {
        i + 1: float(rows.loc[i, "price"]) if i < len(rows) else last_price for i in range(N_D)
    }

    solar: dict[int, float] = {}
    if df_solar is not None and not df_solar.empty and not rows.empty:
        ts_start = rows["timestamp"].iloc[0]
        pv_base = int(df_solar["timestamp_utc"].searchsorted(ts_start))
        for i in range(N_D):
            pv_idx = pv_base + i
            solar[i + 1] = (
                float(df_solar.iloc[pv_idx]["generation_MW"]) if pv_idx < len(df_solar) else 0.0
            )
    else:
        solar = {i + 1: 0.0 for i in range(N_D)}

    return price, solar


# ---------------------------------------------------------------------------
# Predicted prices: call the LGBM model for a 24-h horizon
# ---------------------------------------------------------------------------
def build_predicted_price_dict(
    ts: datetime,
    fallback_price: float = 100.0,
) -> dict[int, float]:
    """
    Run the LGBM price prediction model with *ts* as reference_time and return
    a 96-period {1..96: float} price dict for the optimizer.

    ``use_csv=True`` is used so that features are drawn from the static training
    CSVs in ``Data/`` rather than live APIs, making historical replay
    reproducible and independent of weather/BMRS network calls.

    Falls back to *fallback_price* for all periods if the model call fails.
    """
    try:
        from ors.services.price_inference import run_inference
        from ors.services.price_inference.live_inference import LGBM_MODEL_DIR

        df_pred = run_inference(
            model_path=LGBM_MODEL_DIR,
            reference_time=ts,
            use_csv=True,
            project_root=REPO_ROOT,
        )
        prices = df_pred["Price_pred"].values
        n = len(prices)
        last = float(prices[-1]) if n > 0 else fallback_price
        return {i + 1: float(prices[i]) if i < n else last for i in range(N_D)}
    except Exception as exc:
        print(
            f"  WARNING: price inference failed at {ts}: {exc}. "
            f"Using fallback price {fallback_price:.2f} for all periods."
        )
        return {i + 1: fallback_price for i in range(N_D)}


# ---------------------------------------------------------------------------
# Single optimiser call (with SOC override via module-level patch)
# ---------------------------------------------------------------------------
def run_single_optimize(
    solver,
    price: dict[int, float],
    solar: dict[int, float],
    p_30: float,
    cycles_used_today: int,
    t_boundary: int,
    e0: float,
    q_init: int = 0,
    z_dis_init: int = 0,
):
    """
    Build and solve the MILP with e0 as the initial SOC.
    q_init and z_dis_init carry the pre-horizon cycle state so the optimizer
    can correctly detect a cycle event at t=1.
    Returns (model, solve_info). On failure, model is None.
    """
    battery_spec = types.SimpleNamespace(**vars(_BASE_BATTERY_SPEC))
    battery_spec.current_energy_mwh = e0
    try:
        m = opt_module.build_model(
            price,
            solar,
            p_30,
            cycles_used_today,
            t_boundary,
            q_init=q_init,
            z_dis_init=z_dis_init,
            battery_spec=battery_spec,
        )
        result = solver.solve(m, tee=False)

        status = result.solver.status
        term = result.solver.termination_condition
        if status != SolverStatus.ok or term not in {
            TerminationCondition.optimal,
            TerminationCondition.feasible,
        }:
            return None, {
                "solver_status": str(status),
                "termination_condition": str(term),
                "error": f"solver status={status}, termination={term}",
            }

        return m, {
            "solver_status": str(status),
            "termination_condition": str(term),
            "error": "",
        }
    except Exception as exc:
        return None, {
            "solver_status": "exception",
            "termination_condition": "exception",
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        pass  # E0 is no longer a module-level constant


# ---------------------------------------------------------------------------
# Physics: simulate one 15-min step
# ---------------------------------------------------------------------------
def simulate_step(e_prev: float, p_grid: float, p_dis: float, p_sol_bat: float) -> float:
    """Return the new energy level after one DT-hour step, clamped to [E_MIN, E_MAX]."""
    p_ch = p_grid + p_sol_bat
    e_new = e_prev + ETA_CH * p_ch * DT - (p_dis * DT) / ETA_DIS - P_AUX * DT - e_prev * R_SD * DT
    return float(max(E_MIN, min(E_MAX, e_new)))


def step_profit(price_val: float, p_grid: float, p_dis: float, p_sol_sell: float) -> float:
    """Revenue from discharging/solar-sell minus cost of grid charging, for one DT step."""
    return price_val * (p_dis + ETA_SOL_SELL * p_sol_sell - p_grid) * DT


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------
def run_backtest(
    df_all: pd.DataFrame,
    start_dt: datetime,
    n_sim_days: int,
    commit_periods: int,
    solver,
    df_solar: pd.DataFrame | None = None,
    trace_records: list[dict] | None = None,
    trace_output_path: str | None = None,
    price_source: str = "perfect",
) -> pd.DataFrame:
    """
    Rolling-horizon backtest.

    Parameters
    ----------
    df_all : full historical price DataFrame (UTC-sorted).
    start_dt : first timestamp to simulate (UTC-aware, should be midnight).
    n_sim_days : number of days to simulate.
    commit_periods : number of 15-min periods to commit per optimizer run
                     (16 for 4-hour re-planning, 1 for 15-min re-planning).
    solver : Pyomo solver instance.
    df_solar : pre-generated site PV data (timestamp_utc, generation_MW).
               If None, solar defaults to 0 for all periods.
    trace_records : optional list that receives one debug record per simulated
                    step with optimizer inputs, solve status, and committed/
                    applied actions.
    trace_output_path : optional CSV path for live trace streaming. If set,
                        each step is appended and flushed immediately.
    price_source : ``"perfect"`` — feed true historical prices to the optimizer
                   (perfect-information baseline).
                   ``"predicted"`` — feed LGBM price model output to the
                   optimizer; profit is still settled at true prices.

    Returns
    -------
    DataFrame with one row per 15-min step and columns:
        timestamp, price, P_grid_MW, P_dis_MW, P_sol_bat_MW, P_sol_sell_MW,
        z_grid, z_solbat, z_dis, cycle, E_MWh, profit_step
    """
    idx_start = int(df_all["timestamp"].searchsorted(start_dt))
    total_steps = n_sim_days * N_D

    e_soc = 0.5 * E_CAP  # initial state of charge
    cycles_since_23 = 0  # cycles executed since last 23:00 boundary

    # Simulation-state cycle detection: compensates for the optimizer skipping
    # cycle_and constraints at t=1, which makes m.cycle[1] always 0.
    # In 1-week mode every committed step is t=1, so without this every cycle
    # count would be 0.
    q_sim = False  # True once we have charged since the last discharge
    prev_z_dis_sim = 0  # z_dis value from the previous committed step

    # Current committed plan: list of tuples (one per 15-min period over 24h)
    committed_plan: list[tuple] = []
    committed_from_step: int = -1
    n_optimizer_calls = 0
    n_failed_calls = 0
    plan_source = "none"
    plan_origin_step = -1

    # Snapshot of the most recent optimizer input horizon for step-level tracing.
    input_snapshot = {
        "input_price_t1": None,
        "input_price_t16": None,
        "input_price_t96": None,
        "input_price_min": None,
        "input_price_max": None,
        "input_solar_t1": None,
        "input_solar_t16": None,
        "input_solar_t96": None,
        "input_solar_min": None,
        "input_solar_max": None,
    }

    # Robust 23:00 reset boundary handling for datasets with missing timestamps.
    ts0 = df_all.iloc[idx_start]["timestamp"].to_pydatetime()
    if ts0.hour >= 23:
        next_cycle_reset = (ts0 + timedelta(days=1)).replace(
            hour=23, minute=0, second=0, microsecond=0
        )
    else:
        next_cycle_reset = ts0.replace(hour=23, minute=0, second=0, microsecond=0)

    records: list[dict] = []

    trace_fieldnames = [
        "step",
        "timestamp",
        "need_replan",
        "optimizer_called",
        "plan_source",
        "plan_origin_step",
        "committed_offset",
        "data_idx",
        "actual_price",
        "e_soc_before",
        "e_soc_after",
        "cycles_since_23_before",
        "cycles_since_23_after",
        "cycles_for_model",
        "t_boundary",
        "p_30",
        "solver_status",
        "termination_condition",
        "fail_reason",
        "plan0_p_grid",
        "plan0_p_dis",
        "plan0_p_sol_bat",
        "plan0_p_sol_sell",
        "plan0_z_dis",
        "plan0_z_grid",
        "plan0_z_solbat",
        "plan0_cycle",
        "applied_p_grid",
        "applied_p_dis",
        "applied_p_sol_bat",
        "applied_p_sol_sell",
        "applied_z_dis",
        "applied_z_grid",
        "applied_z_solbat",
        "applied_cycle",
        "input_price_t1",
        "input_price_t16",
        "input_price_t96",
        "input_price_min",
        "input_price_max",
        "input_solar_t1",
        "input_solar_t16",
        "input_solar_t96",
        "input_solar_min",
        "input_solar_max",
    ]

    trace_file = None
    trace_writer = None
    if trace_output_path is not None:
        trace_file = open(trace_output_path, "w", newline="", encoding="utf-8")  # noqa: SIM115
        trace_writer = csv.DictWriter(trace_file, fieldnames=trace_fieldnames)
        trace_writer.writeheader()
        trace_file.flush()

    try:
        for step in range(total_steps):
            data_idx = idx_start + step

            # Guard: need N_D look-ahead; if we run out of data, stop
            if data_idx + N_D > len(df_all):
                print(f"  [step {step}] Reached end of price data. Stopping early.")
                break

            ts: pd.Timestamp = df_all.iloc[data_idx]["timestamp"]
            actual_price = float(df_all.iloc[data_idx]["price"])

            # ------------------------------------------------------------------
            # Re-plan?  Yes when we have no plan or the current commit window
            # has been consumed.
            # ------------------------------------------------------------------
            need_replan = len(committed_plan) == 0 or (step - committed_from_step) >= commit_periods

            if need_replan:
                price_dict_perfect, solar_dict = build_price_solar_dicts(df_all, data_idx, df_solar)
                if price_source == "predicted":
                    price_dict = build_predicted_price_dict(ts.to_pydatetime())
                else:
                    price_dict = price_dict_perfect
                p_30 = compute_p30(df_all, ts.to_pydatetime())

                input_snapshot = {
                    "input_price_t1": float(price_dict[1]),
                    "input_price_t16": float(price_dict[min(16, N_D)]),
                    "input_price_t96": float(price_dict[N_D]),
                    "input_price_min": float(min(price_dict.values())),
                    "input_price_max": float(max(price_dict.values())),
                    "input_solar_t1": float(solar_dict[1]),
                    "input_solar_t16": float(solar_dict[min(16, N_D)]),
                    "input_solar_t96": float(solar_dict[N_D]),
                    "input_solar_min": float(min(solar_dict.values())),
                    "input_solar_max": float(max(solar_dict.values())),
                }

                # Periods until the next 23:00 (optimizer's daily-cycle boundary)
                ts_py = ts.to_pydatetime()
                if ts_py.hour >= 23:
                    next_23 = (ts_py + timedelta(days=1)).replace(
                        hour=23, minute=0, second=0, microsecond=0
                    )
                else:
                    next_23 = ts_py.replace(hour=23, minute=0, second=0, microsecond=0)
                hours_to_23 = (next_23 - ts_py).total_seconds() / 3600.0
                t_boundary = max(1, min(N_D, int(round(hours_to_23 / DT))))

                cycles_for_model = min(cycles_since_23, MAX_CYCLES_PER_DAY)
                if cycles_since_23 > MAX_CYCLES_PER_DAY:
                    print(
                        f"  [step {step}] WARNING: cycles_since_23={cycles_since_23} exceeds "
                        f"MAX_CYCLES_PER_DAY={MAX_CYCLES_PER_DAY}. Clamping to avoid infeasible MILP."
                    )

                n_optimizer_calls += 1
                m, solve_info = run_single_optimize(
                    solver,
                    price_dict,
                    solar_dict,
                    p_30,
                    cycles_for_model,
                    t_boundary,
                    e_soc,
                    q_init=int(q_sim),
                    z_dis_init=prev_z_dis_sim,
                )
                fail_reason = solve_info["error"]

                if m is not None:
                    try:
                        committed_plan = [
                            (
                                float(value(m.P_grid[t])),
                                float(value(m.P_dis[t])),
                                float(value(m.P_sol_bat[t])),
                                float(value(m.P_sol_sell[t])),
                                int(round(float(value(m.z_dis[t])))),
                                int(round(float(value(m.z_grid[t])))),
                                int(round(float(value(m.z_solbat[t])))),
                                int(round(float(value(m.cycle[t])))),
                            )
                            for t in m.T
                        ]
                        plan_source = "optimized"
                    except Exception:
                        m = None  # fall through to idle plan
                        fail_reason = "could not read solved variable values"
                        solve_info = {
                            "solver_status": "read_error",
                            "termination_condition": "read_error",
                            "error": fail_reason,
                        }

                if m is None:
                    n_failed_calls += 1
                    print(
                        f"  [step {step}] Optimizer failed at {ts}: {fail_reason}. "
                        "Applying idle fallback plan."
                    )
                    # Idle (no charge/discharge) for all 96 periods
                    committed_plan = [(0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0)] * N_D
                    plan_source = "idle_fallback"

                committed_from_step = step
                plan_origin_step = step
            else:
                solve_info = {
                    "solver_status": "not_called",
                    "termination_condition": "not_called",
                    "error": "",
                }
                p_30 = None
                t_boundary = None
                cycles_for_model = None

            # ------------------------------------------------------------------
            # Apply committed action for this step
            # ------------------------------------------------------------------
            e_before = e_soc
            cycles_before = cycles_since_23
            offset = step - committed_from_step
            p_grid, p_dis, p_sol_bat, p_sol_sell, z_dis, z_grid, z_solbat, _cyc_model = (
                committed_plan[offset]
            )

            # Simulation-based cycle detection: the optimizer skips cycle_and
            # constraints at t=1, so m.cycle[1] is always 0.  In 1-week mode
            # every committed step is t=1, meaning _cyc_model is always 0.
            # We reproduce the same cycle logic (start-of-discharge after
            # charging) from the actual applied z_dis / z_grid / z_solbat flags.
            s_dis_now = int(z_dis == 1 and prev_z_dis_sim == 0)
            cyc = int(s_dis_now and q_sim)
            # Update charge flag: set on any charging step, clear when discharging.
            if z_dis:
                q_sim = False
            elif z_grid or z_solbat:
                q_sim = True
            prev_z_dis_sim = z_dis

            # Simulate actual SOC transition
            e_new = simulate_step(e_soc, p_grid, p_dis, p_sol_bat)
            profit = step_profit(actual_price, p_grid, p_dis, p_sol_sell)

            # Reset by boundary crossing (works even if exact 23:00 row is missing).
            ts_py = ts.to_pydatetime()
            if ts_py >= next_cycle_reset:
                cycles_since_23 = 0
                while ts_py >= next_cycle_reset:
                    next_cycle_reset += timedelta(days=1)
            if cyc:
                cycles_since_23 += 1

            cycles_after = cycles_since_23

            plan_first = committed_plan[0] if committed_plan else (None,) * 8
            trace_row = {
                "step": step,
                "timestamp": ts,
                "need_replan": int(need_replan),
                "optimizer_called": int(need_replan),
                "plan_source": plan_source,
                "plan_origin_step": plan_origin_step,
                "committed_offset": offset,
                "data_idx": data_idx,
                "actual_price": actual_price,
                "e_soc_before": e_before,
                "e_soc_after": e_new,
                "cycles_since_23_before": cycles_before,
                "cycles_since_23_after": cycles_after,
                "cycles_for_model": cycles_for_model,
                "t_boundary": t_boundary,
                "p_30": p_30,
                "solver_status": solve_info["solver_status"],
                "termination_condition": solve_info["termination_condition"],
                "fail_reason": solve_info["error"],
                "plan0_p_grid": plan_first[0] if committed_plan else None,
                "plan0_p_dis": plan_first[1] if committed_plan else None,
                "plan0_p_sol_bat": plan_first[2] if committed_plan else None,
                "plan0_p_sol_sell": plan_first[3] if committed_plan else None,
                "plan0_z_dis": plan_first[4] if committed_plan else None,
                "plan0_z_grid": plan_first[5] if committed_plan else None,
                "plan0_z_solbat": plan_first[6] if committed_plan else None,
                "plan0_cycle": plan_first[7] if committed_plan else None,
                "applied_p_grid": p_grid,
                "applied_p_dis": p_dis,
                "applied_p_sol_bat": p_sol_bat,
                "applied_p_sol_sell": p_sol_sell,
                "applied_z_dis": z_dis,
                "applied_z_grid": z_grid,
                "applied_z_solbat": z_solbat,
                "applied_cycle": cyc,
                **input_snapshot,
            }

            if trace_records is not None:
                trace_records.append(trace_row)
            if trace_writer is not None and trace_file is not None:
                trace_writer.writerow(trace_row)
                trace_file.flush()

            records.append(
                {
                    "timestamp": ts,
                    "price": actual_price,
                    "P_grid_MW": p_grid,
                    "P_dis_MW": p_dis,
                    "P_sol_bat_MW": p_sol_bat,
                    "P_sol_sell_MW": p_sol_sell,
                    "z_grid": z_grid,
                    "z_solbat": z_solbat,
                    "z_dis": z_dis,
                    "cycle": cyc,
                    "E_MWh": e_new,
                    "profit_step": profit,
                }
            )
            e_soc = e_new

            # Progress log: once per day
            if step % N_D == 0:
                day_num = step // N_D + 1
                print(
                    f"  Day {day_num:>4d}/{n_sim_days}  |  "
                    f"ts={ts.date()}  |  "
                    f"SOC={e_soc:.1f} MWh  |  "
                    f"optimizer calls so far: {n_optimizer_calls}"
                )
    finally:
        if trace_file is not None:
            trace_file.close()

    print(f"\nOptimizer calls: {n_optimizer_calls} total, {n_failed_calls} failed.")
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Daily summary
# ---------------------------------------------------------------------------
def daily_summary(df_results: pd.DataFrame) -> pd.DataFrame:
    df = df_results.copy()
    df["date"] = df["timestamp"].dt.date
    summary = (
        df.groupby("date")
        .agg(
            profit_GBP=("profit_step", "sum"),
            n_cycles=("cycle", "sum"),
            avg_price_GBP_MWh=("price", "mean"),
            min_SOC_MWh=("E_MWh", "min"),
            max_SOC_MWh=("E_MWh", "max"),
        )
        .reset_index()
        .rename(columns={"date": "Date"})
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="BESS Optimizer Backtester — perfect-information rolling horizon"
    )
    parser.add_argument(
        "--mode",
        choices=["4month", "1week"],
        required=True,
        help="4month: re-plan every 4 h over 120 days. 1week: re-plan every 15 min over 7 days.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for window selection")
    parser.add_argument(
        "--price-csv",
        default=str(PRICE_CSV),
        help="Path to price_data_rotated_2d.csv",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path for detailed per-step results (optional)",
    )
    parser.add_argument(
        "--solar-chunk-days",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Number of days per PV generation request (default 1). "
            "Increase to reduce API calls at the cost of potentially wider "
            "forecast horizons per request."
        ),
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Write per-step debug trace with optimizer inputs/status/decisions.",
    )
    parser.add_argument(
        "--trace-output",
        default=None,
        help="Output CSV path for trace (defaults to <detail>_trace.csv when --trace is used).",
    )
    parser.add_argument(
        "--price-source",
        choices=["perfect", "predicted"],
        default="perfect",
        help=(
            "'perfect': feed true historical prices to the optimizer (perfect-information "
            "baseline). 'predicted': feed LGBM model output as the optimizer forecast; "
            "profit is still settled at true prices. Run both with the same --seed to "
            "compare value of perfect information."
        ),
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # --- Load price data ---
    print(f"Loading price data: {args.price_csv}")
    df_all = load_price_data(Path(args.price_csv))
    ts_min = df_all["timestamp"].min()
    ts_max = df_all["timestamp"].max()
    print(f"  {len(df_all)} rows  |  {ts_min.date()} -> {ts_max.date()}")

    # --- Mode parameters ---
    if args.mode == "4month":
        n_days = 120
        commit_periods = PERIODS_PER_4H
        label = f"4-month (120 days), re-plan every 4 hours [{args.price_source} prices]"
    else:
        n_days = 7
        commit_periods = PERIODS_PER_15MIN
        label = f"1-week (7 days), re-plan every 15 minutes [{args.price_source} prices]"

    # --- Choose random window ---
    # Need HIST_DAYS of prior data and n_days + 1 day of ahead data.
    min_prior_rows = HIST_DAYS * N_D
    n_sim_rows = n_days * N_D

    # Collect candidate indices that fall on midnight UTC
    candidates = [
        i
        for i in range(min_prior_rows, len(df_all) - n_sim_rows - N_D)
        if df_all.iloc[i]["timestamp"].hour == 0 and df_all.iloc[i]["timestamp"].minute == 0
    ]

    if not candidates:
        print("ERROR: Not enough data in the CSV for the selected mode.")
        sys.exit(1)

    start_idx = rng.choice(candidates)
    start_dt: datetime = df_all.iloc[start_idx]["timestamp"].to_pydatetime()
    end_dt = start_dt + timedelta(days=n_days)

    print(f"\nMode        : {label}")
    print(f"Window      : {start_dt.date()} -> {end_dt.date()}")
    print(f"Seed        : {args.seed}")
    if args.price_source == "predicted":
        print("Price source: LGBM model predictions (use_csv=True) — profit settled at true prices")

    # --- Solver ---
    solver = get_solver()

    # --- Expected optimizer calls ---
    expected_calls = n_days * (24 // 4) if args.mode == "4month" else n_days * N_D

    print(f"\nExpected optimizer calls: ~{expected_calls}")
    print(f"Running {n_days} days x {N_D} steps/day = {n_days * N_D} total steps")

    # --- Pre-generate solar for the whole window (once, not per optimizer call) ---
    print("\nPre-generating PV solar forecast for simulation window...")
    df_solar = pregenerate_solar(start_dt, end_dt)

    print("\nStarting backtest loop...\n")

    trace_records: list[dict] | None = None
    trace_path = None
    if args.trace:
        if args.output is None:
            seed_tag = args.seed if args.seed is not None else "rand"
            out_dir = REPO_ROOT / "tests" / "backtesting"
            detail_preview = str(
                out_dir / f"backtest_detail_{args.mode}_{args.price_source}_{seed_tag}.csv"
            )
        else:
            detail_preview = args.output
        trace_path = args.trace_output or detail_preview.replace(".csv", "_trace.csv")
        print(f"Live trace enabled : {trace_path}")

    # --- Run ---
    df_results = run_backtest(
        df_all=df_all,
        start_dt=start_dt,
        n_sim_days=n_days,
        commit_periods=commit_periods,
        solver=solver,
        df_solar=df_solar,
        trace_records=trace_records,
        trace_output_path=trace_path,
        price_source=args.price_source,
    )

    if df_results.empty:
        print("No results produced.")
        sys.exit(1)

    # --- Daily summary ---
    daily = daily_summary(df_results)

    print("\n" + "=" * 72)
    print(f"DAILY PROFIT SUMMARY  ({label})")
    print("=" * 72)
    print(
        daily.to_string(
            index=False,
            float_format=lambda x: f"{x:,.2f}",
        )
    )

    total_profit = daily["profit_GBP"].sum()
    actual_days = len(daily)
    print("-" * 72)
    print(f"Total profit  : £{total_profit:>12,.2f}")
    print(f"Days simulated: {actual_days}")
    print(f"Average/day   : £{total_profit / actual_days:>12,.2f}")
    print("=" * 72)

    # --- Save outputs ---
    if args.output is None:
        seed_tag = args.seed if args.seed is not None else "rand"
        out_dir = REPO_ROOT / "tests" / "backtesting"
        detail_path = str(
            out_dir / f"backtest_detail_{args.mode}_{args.price_source}_{seed_tag}.csv"
        )
        summary_path = str(
            out_dir / f"backtest_summary_{args.mode}_{args.price_source}_{seed_tag}.csv"
        )
    else:
        detail_path = args.output
        summary_path = detail_path.replace(".csv", "_summary.csv")

    df_results.to_csv(detail_path, index=False)
    daily.to_csv(summary_path, index=False)

    print(f"\nDetailed results : {detail_path}")
    print(f"Daily summary    : {summary_path}")
    if trace_path is not None:
        print(f"Trace output     : {trace_path}")


if __name__ == "__main__":
    main()
