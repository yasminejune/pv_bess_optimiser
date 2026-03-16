"""Unit tests for tests/backtesting/backtest.py.

Categories
----------
TestLoadPriceData       : load_price_data() — csv loading, timestamp parsing, sorting
TestComputeP30          : compute_p30() — 30-day rolling early-morning average
TestBuildDicts          : build_price_solar_dicts() — optimizer input construction
TestBuildPredictedPrice : build_predicted_price_dict() — LGBM wrapper + fallback
TestSimulateStep        : simulate_step() — per-step battery physics
TestStepProfit          : step_profit() — per-step revenue/cost accounting
TestRunSingleOptimize   : run_single_optimize() — solver call wrapper (mocked solver)
TestDailySummary        : daily_summary() — per-day aggregation
TestRunBacktest         : run_backtest() — rolling-horizon loop (solver mocked)
"""

from __future__ import annotations

import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pyomo.opt import SolverStatus, TerminationCondition

# ---------------------------------------------------------------------------
# Load the backtest module from its file path so as to avoid name-collision
# with the ``backtest`` folder itself.
# ---------------------------------------------------------------------------
_BT_PATH = Path(__file__).resolve().parent / "backtest.py"
_spec = importlib.util.spec_from_file_location("_backtest_module", _BT_PATH)
bt = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(bt)  # type: ignore[union-attr]

# Re-bind names for brevity in tests
load_price_data = bt.load_price_data
compute_p30 = bt.compute_p30
build_price_solar_dicts = bt.build_price_solar_dicts
build_predicted_price_dict = bt.build_predicted_price_dict
run_single_optimize = bt.run_single_optimize
simulate_step = bt.simulate_step
step_profit = bt.step_profit
daily_summary = bt.daily_summary
run_backtest = bt.run_backtest

N_D = bt.N_D
DT = bt.DT
E_CAP = bt.E_CAP
E_MIN = bt.E_MIN
E_MAX = bt.E_MAX
ETA_CH = bt.ETA_CH
ETA_DIS = bt.ETA_DIS
ETA_SOL_SELL = bt.ETA_SOL_SELL
P_AUX = bt.P_AUX
R_SD = bt.R_SD
H_30 = bt.H_30
HIST_DAYS = bt.HIST_DAYS


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_price_df(n_rows: int = 300, base_price: float = 100.0) -> pd.DataFrame:
    """Synthetic UTC-aware price DataFrame at 15-min resolution."""
    ts = pd.date_range("2020-01-01", periods=n_rows, freq="15min", tz="UTC")
    return pd.DataFrame({"timestamp": ts, "price": [base_price] * n_rows})


def _make_solar_df(n_rows: int = 300, gen_mw: float = 5.0) -> pd.DataFrame:
    """Synthetic solar generation DataFrame at 15-min resolution."""
    ts = pd.date_range("2020-01-01", periods=n_rows, freq="15min", tz="UTC")
    return pd.DataFrame({"timestamp_utc": ts, "generation_MW": [gen_mw] * n_rows})


def _idle_optimizer(*args, **kwargs):
    """Fake optimizer that always forces the idle-fallback path."""
    return None, {"solver_status": "ok", "termination_condition": "feasible", "error": ""}


# ---------------------------------------------------------------------------
# TestLoadPriceData
# ---------------------------------------------------------------------------


class TestLoadPriceData:
    def test_returns_dataframe_with_required_columns(self, tmp_path):
        csv = tmp_path / "prices.csv"
        csv.write_text(
            "timestamp,price\n"
            "2020-01-01 00:00:00+00:00,100.0\n"
            "2020-01-01 00:15:00+00:00,110.0\n"
        )
        df = load_price_data(csv)
        assert "timestamp" in df.columns
        assert "price" in df.columns

    def test_timestamps_are_utc_aware(self, tmp_path):
        csv = tmp_path / "prices.csv"
        csv.write_text("timestamp,price\n2020-01-01 00:00:00+00:00,100.0\n")
        df = load_price_data(csv)
        assert df["timestamp"].dt.tz is not None

    def test_result_is_sorted_ascending(self, tmp_path):
        csv = tmp_path / "prices.csv"
        csv.write_text(
            "timestamp,price\n"
            "2020-01-01 01:00:00+00:00,120.0\n"
            "2020-01-01 00:00:00+00:00,100.0\n"
            "2020-01-01 00:30:00+00:00,110.0\n"
        )
        df = load_price_data(csv)
        ts = df["timestamp"].values
        assert all(ts[i] <= ts[i + 1] for i in range(len(ts) - 1))

    def test_prices_are_preserved(self, tmp_path):
        csv = tmp_path / "prices.csv"
        csv.write_text("timestamp,price\n2020-01-01 00:00:00+00:00,99.5\n")
        df = load_price_data(csv)
        assert df["price"].iloc[0] == pytest.approx(99.5)

    def test_reset_index_after_sort(self, tmp_path):
        csv = tmp_path / "prices.csv"
        csv.write_text(
            "timestamp,price\n"
            "2020-01-01 00:30:00+00:00,200.0\n"
            "2020-01-01 00:00:00+00:00,100.0\n"
        )
        df = load_price_data(csv)
        # Index should be 0-based after reset
        assert df.index.tolist() == [0, 1]
        assert df["price"].iloc[0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# TestComputeP30
# ---------------------------------------------------------------------------


class TestComputeP30:
    def _make_hist_df(
        self, base_dt: datetime, n_days: int = 35, early_price: float = 80.0
    ) -> pd.DataFrame:
        """DataFrame with `early_price` in the H_30 morning window, 120 elsewhere."""
        start = base_dt - timedelta(days=n_days)
        ts = pd.date_range(start, periods=n_days * N_D, freq="15min", tz="UTC")
        prices = [
            early_price if (t.hour + t.minute / 60.0) < H_30 else 120.0 for t in ts
        ]
        return pd.DataFrame({"timestamp": ts, "price": prices})

    def test_returns_float(self):
        run_time = datetime(2020, 3, 1, 12, 0, tzinfo=timezone.utc)
        df = self._make_hist_df(run_time)
        assert isinstance(compute_p30(df, run_time), float)

    def test_uses_early_morning_prices(self):
        run_time = datetime(2020, 3, 1, 12, 0, tzinfo=timezone.utc)
        df = self._make_hist_df(run_time, early_price=60.0)
        result = compute_p30(df, run_time)
        # Early-morning average should be ~60, not the 120 rest-of-day price
        assert result < 100.0

    def test_falls_back_to_mean_when_no_history(self):
        # run_time at the very start of the DataFrame — no prior 30-day window
        run_time = datetime(2020, 1, 1, tzinfo=timezone.utc)
        df = _make_price_df(200, base_price=50.0)
        result = compute_p30(df, run_time)
        assert result == pytest.approx(50.0)

    def test_earlier_vs_later_run_time_same_df(self):
        # Moving run_time later gives more history — result should still be a float
        df = _make_hist_df = self._make_hist_df
        run_early = datetime(2020, 2, 1, tzinfo=timezone.utc)
        run_late = datetime(2020, 3, 1, tzinfo=timezone.utc)
        df = self._make_hist_df(run_late, n_days=60)
        r_early = compute_p30(df, run_early)
        r_late = compute_p30(df, run_late)
        assert isinstance(r_early, float)
        assert isinstance(r_late, float)


# ---------------------------------------------------------------------------
# TestBuildDicts
# ---------------------------------------------------------------------------


class TestBuildDicts:
    def test_no_solar_returns_all_zeros(self):
        df = _make_price_df(200)
        _, solar = build_price_solar_dicts(df, 0, None)
        assert all(v == 0.0 for v in solar.values())
        assert len(solar) == N_D

    def test_dict_keys_are_1_to_96(self):
        df = _make_price_df(200)
        price, solar = build_price_solar_dicts(df, 0, None)
        assert set(price.keys()) == set(range(1, N_D + 1))
        assert set(solar.keys()) == set(range(1, N_D + 1))

    def test_price_values_match_source(self):
        df = _make_price_df(200, base_price=77.0)
        price, _ = build_price_solar_dicts(df, 0, None)
        assert all(v == pytest.approx(77.0) for v in price.values())

    def test_start_idx_offsets_into_dataframe(self):
        df = _make_price_df(300, base_price=50.0)
        df.loc[100, "price"] = 999.0  # first price at start_idx=100
        price, _ = build_price_solar_dicts(df, 100, None)
        assert price[1] == pytest.approx(999.0)

    def test_solar_aligned_to_price_timestamps(self):
        df_price = _make_price_df(200, base_price=50.0)
        df_solar = _make_solar_df(200, gen_mw=8.0)
        _, solar = build_price_solar_dicts(df_price, 0, df_solar)
        assert all(solar[t] == pytest.approx(8.0) for t in range(1, N_D + 1))

    def test_tail_padding_uses_last_available_price(self):
        # Only 20 rows available — indices 20..95 should be padded with row-19's price
        df = _make_price_df(20, base_price=50.0)
        df.loc[19, "price"] = 999.0  # last available row
        price, _ = build_price_solar_dicts(df, 0, None)
        # Padded periods (t > 20) should all equal 999.0
        assert price[N_D] == pytest.approx(999.0)
        assert price[21] == pytest.approx(999.0)

    def test_empty_solar_df_falls_back_to_zeros(self):
        df_price = _make_price_df(200)
        df_solar = pd.DataFrame(columns=["timestamp_utc", "generation_MW"])
        _, solar = build_price_solar_dicts(df_price, 0, df_solar)
        assert all(v == 0.0 for v in solar.values())


# ---------------------------------------------------------------------------
# TestBuildPredictedPrice
# ---------------------------------------------------------------------------


class TestBuildPredictedPrice:
    def test_returns_n_d_values(self):
        ts = datetime(2022, 6, 1, 12, 0, tzinfo=timezone.utc)
        result = build_predicted_price_dict(ts, fallback_price=99.0)
        assert len(result) == N_D
        assert set(result.keys()) == set(range(1, N_D + 1))

    def test_values_are_floats(self):
        ts = datetime(2022, 6, 1, 12, 0, tzinfo=timezone.utc)
        result = build_predicted_price_dict(ts, fallback_price=55.0)
        assert all(isinstance(v, float) for v in result.values())

    def test_fallback_when_run_inference_raises(self):
        ts = datetime(2022, 6, 1, tzinfo=timezone.utc)
        mock_ri = MagicMock(side_effect=RuntimeError("model unavailable"))
        mock_li = MagicMock()
        mock_li.LGBM_MODEL_DIR = Path("/tmp/nonexistent_model")
        with patch.dict(
            "sys.modules",
            {
                "ors.services.price_inference": MagicMock(run_inference=mock_ri),
                "ors.services.price_inference.live_inference": mock_li,
            },
        ):
            result = build_predicted_price_dict(ts, fallback_price=42.0)
        assert all(v == pytest.approx(42.0) for v in result.values())
        assert len(result) == N_D

    def test_fallback_value_is_respected(self):
        ts = datetime(2022, 6, 1, tzinfo=timezone.utc)
        mock_module = MagicMock()
        mock_module.run_inference.side_effect = ValueError("bad input")
        with patch.dict(
            "sys.modules",
            {
                "ors.services.price_inference": mock_module,
                "ors.services.price_inference.live_inference": MagicMock(
                    LGBM_MODEL_DIR=Path("/tmp")
                ),
            },
        ):
            result = build_predicted_price_dict(ts, fallback_price=123.456)
        assert all(v == pytest.approx(123.456) for v in result.values())


# ---------------------------------------------------------------------------
# TestSimulateStep
# ---------------------------------------------------------------------------


class TestSimulateStep:
    def test_returns_float(self):
        e_new = simulate_step(E_CAP * 0.5, p_grid=0.0, p_dis=0.0, p_sol_bat=0.0)
        assert isinstance(e_new, float)

    def test_idle_step_decreases_soc_by_losses(self):
        e0 = E_CAP * 0.5
        e_new = simulate_step(e0, p_grid=0.0, p_dis=0.0, p_sol_bat=0.0)
        expected = e0 - P_AUX * DT - e0 * R_SD * DT
        assert e_new == pytest.approx(max(E_MIN, min(E_MAX, expected)), abs=1e-6)

    def test_charging_increases_soc(self):
        e0 = E_MIN + 20.0
        e_new = simulate_step(e0, p_grid=50.0, p_dis=0.0, p_sol_bat=0.0)
        assert e_new > e0

    def test_discharging_decreases_soc(self):
        e0 = E_CAP * 0.8
        e_new = simulate_step(e0, p_grid=0.0, p_dis=50.0, p_sol_bat=0.0)
        assert e_new < e0

    def test_soc_clamped_to_e_min(self):
        # Large discharge on near-empty battery must not go below E_MIN
        e_new = simulate_step(E_MIN, p_grid=0.0, p_dis=200.0, p_sol_bat=0.0)
        assert e_new >= E_MIN - 1e-9

    def test_soc_clamped_to_e_max(self):
        # Large charge on near-full battery must not exceed E_MAX
        e_new = simulate_step(E_MAX - 1.0, p_grid=200.0, p_dis=0.0, p_sol_bat=0.0)
        assert e_new <= E_MAX + 1e-9

    def test_solar_bat_contribution_raises_soc(self):
        e0 = E_MIN + 30.0
        e_no_sol = simulate_step(e0, p_grid=0.0, p_dis=0.0, p_sol_bat=0.0)
        e_with_sol = simulate_step(e0, p_grid=0.0, p_dis=0.0, p_sol_bat=20.0)
        assert e_with_sol > e_no_sol

    def test_discharge_physics_formula(self):
        e0 = 200.0
        p_dis = 30.0
        expected = e0 - (p_dis * DT) / ETA_DIS - P_AUX * DT - e0 * R_SD * DT
        expected = max(E_MIN, min(E_MAX, expected))
        e_new = simulate_step(e0, p_grid=0.0, p_dis=p_dis, p_sol_bat=0.0)
        assert e_new == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# TestStepProfit
# ---------------------------------------------------------------------------


class TestStepProfit:
    def test_idle_yields_zero_profit(self):
        profit = step_profit(price_val=100.0, p_grid=0.0, p_dis=0.0, p_sol_sell=0.0)
        assert profit == pytest.approx(0.0)

    def test_discharging_gives_positive_profit(self):
        profit = step_profit(price_val=200.0, p_grid=0.0, p_dis=50.0, p_sol_sell=0.0)
        assert profit > 0.0

    def test_grid_charging_gives_negative_profit(self):
        profit = step_profit(price_val=50.0, p_grid=30.0, p_dis=0.0, p_sol_sell=0.0)
        assert profit < 0.0

    def test_solar_sell_adds_to_profit(self):
        p_no_solar = step_profit(100.0, p_grid=0.0, p_dis=10.0, p_sol_sell=0.0)
        p_with_solar = step_profit(100.0, p_grid=0.0, p_dis=10.0, p_sol_sell=20.0)
        assert p_with_solar > p_no_solar

    def test_profit_formula_discharge_only(self):
        price, p_dis = 100.0, 20.0
        expected = price * p_dis * DT
        profit = step_profit(price_val=price, p_grid=0.0, p_dis=p_dis, p_sol_sell=0.0)
        assert profit == pytest.approx(expected, rel=1e-6)

    def test_profit_formula_combined(self):
        price, p_grid, p_dis, p_sol_sell = 100.0, 5.0, 20.0, 10.0
        expected = price * (p_dis + ETA_SOL_SELL * p_sol_sell - p_grid) * DT
        profit = step_profit(price, p_grid=p_grid, p_dis=p_dis, p_sol_sell=p_sol_sell)
        assert profit == pytest.approx(expected, rel=1e-6)

    def test_higher_price_proportional_profit(self):
        p1 = step_profit(price_val=100.0, p_grid=0.0, p_dis=10.0, p_sol_sell=0.0)
        p2 = step_profit(price_val=200.0, p_grid=0.0, p_dis=10.0, p_sol_sell=0.0)
        assert p2 == pytest.approx(2.0 * p1, rel=1e-6)


# ---------------------------------------------------------------------------
# TestRunSingleOptimize
# ---------------------------------------------------------------------------


class TestRunSingleOptimize:
    def _make_inputs(self):
        price = {i: 100.0 for i in range(1, N_D + 1)}
        solar = {i: 0.0 for i in range(1, N_D + 1)}
        return price, solar

    def test_returns_none_model_on_infeasible_status(self):
        mock_solver = MagicMock()
        mock_result = MagicMock()
        mock_result.solver.status = SolverStatus.aborted
        mock_result.solver.termination_condition = TerminationCondition.infeasible
        mock_solver.solve.return_value = mock_result
        price, solar = self._make_inputs()

        m, info = run_single_optimize(
            mock_solver, price, solar, p_30=80.0,
            cycles_used_today=0, t_boundary=96, e0=E_CAP * 0.5,
        )

        assert m is None
        assert info["error"] != ""

    def test_solve_info_contains_expected_keys(self):
        mock_solver = MagicMock()
        mock_result = MagicMock()
        mock_result.solver.status = SolverStatus.aborted
        mock_result.solver.termination_condition = TerminationCondition.infeasible
        mock_solver.solve.return_value = mock_result
        price, solar = self._make_inputs()

        _, info = run_single_optimize(
            mock_solver, price, solar, p_30=80.0,
            cycles_used_today=0, t_boundary=96, e0=E_CAP * 0.5,
        )

        assert "solver_status" in info
        assert "termination_condition" in info
        assert "error" in info

    def test_e0_patch_is_restored_after_call(self):
        """E0 in the optimizer module must be restored regardless of solver outcome."""
        import ors.services.optimizer.optimizer as opt_mod

        original_e0 = opt_mod.E0
        mock_solver = MagicMock()
        mock_solver.solve.side_effect = RuntimeError("crash during solve")
        price, solar = self._make_inputs()

        try:
            run_single_optimize(
                mock_solver, price, solar, p_30=80.0,
                cycles_used_today=0, t_boundary=96, e0=E_CAP * 0.5,
            )
        except Exception:
            pass

        assert opt_mod.E0 == pytest.approx(original_e0)

    def test_exception_in_solver_returns_none(self):
        mock_solver = MagicMock()
        mock_solver.solve.side_effect = RuntimeError("unexpected solver crash")
        price, solar = self._make_inputs()

        m, info = run_single_optimize(
            mock_solver, price, solar, p_30=80.0,
            cycles_used_today=0, t_boundary=96, e0=E_CAP * 0.5,
        )

        assert m is None
        assert "RuntimeError" in info["error"]


# ---------------------------------------------------------------------------
# TestDailySummary
# ---------------------------------------------------------------------------


def _make_results_df(n_days: int = 2, profit_per_step: float = 1.0) -> pd.DataFrame:
    ts = pd.date_range("2020-01-01", periods=n_days * N_D, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "price": [50.0] * (n_days * N_D),
            "P_grid_MW": [0.0] * (n_days * N_D),
            "P_dis_MW": [0.0] * (n_days * N_D),
            "P_sol_bat_MW": [0.0] * (n_days * N_D),
            "P_sol_sell_MW": [0.0] * (n_days * N_D),
            "z_grid": [0] * (n_days * N_D),
            "z_solbat": [0] * (n_days * N_D),
            "z_dis": [0] * (n_days * N_D),
            "cycle": [1] * (n_days * N_D),
            "E_MWh": [E_CAP * 0.5] * (n_days * N_D),
            "profit_step": [profit_per_step] * (n_days * N_D),
        }
    )


class TestDailySummary:
    def test_one_row_per_day(self):
        df = _make_results_df(n_days=3)
        summary = daily_summary(df)
        assert len(summary) == 3

    def test_required_columns_present(self):
        df = _make_results_df(n_days=1)
        summary = daily_summary(df)
        for col in ["Date", "profit_GBP", "n_cycles", "avg_price_GBP_MWh", "min_SOC_MWh", "max_SOC_MWh"]:
            assert col in summary.columns

    def test_daily_profit_sums_step_profits(self):
        per_step = 2.5
        df = _make_results_df(n_days=1, profit_per_step=per_step)
        summary = daily_summary(df)
        assert summary["profit_GBP"].iloc[0] == pytest.approx(per_step * N_D, rel=1e-6)

    def test_n_cycles_counts_cycle_flags(self):
        # All steps have cycle=1 → n_cycles should equal N_D per day
        df = _make_results_df(n_days=1)
        summary = daily_summary(df)
        assert summary["n_cycles"].iloc[0] == N_D

    def test_avg_price_is_mean_of_prices(self):
        df = _make_results_df(n_days=1)
        df["price"] = 75.0
        summary = daily_summary(df)
        assert summary["avg_price_GBP_MWh"].iloc[0] == pytest.approx(75.0)

    def test_min_max_soc_captured(self):
        df = _make_results_df(n_days=1)
        df["E_MWh"] = list(range(N_D))  # increasing from 0 to 95
        summary = daily_summary(df)
        assert summary["min_SOC_MWh"].iloc[0] == pytest.approx(0.0)
        assert summary["max_SOC_MWh"].iloc[0] == pytest.approx(N_D - 1)


# ---------------------------------------------------------------------------
# TestRunBacktest
# ---------------------------------------------------------------------------


class TestRunBacktest:
    """Tests for the rolling-horizon backtest loop with a mocked optimizer."""

    def _minimal_df(self, n_rows: int = 300) -> pd.DataFrame:
        """Enough rows for 1 simulation day + N_D look-ahead."""
        return _make_price_df(n_rows, base_price=100.0)

    def test_output_has_required_columns(self):
        df = self._minimal_df()
        start_dt = df["timestamp"].iloc[0].to_pydatetime()
        with patch.object(bt, "run_single_optimize", side_effect=_idle_optimizer):
            result = run_backtest(df, start_dt, n_sim_days=1, commit_periods=16, solver=None)
        expected_cols = {
            "timestamp", "price", "P_grid_MW", "P_dis_MW", "P_sol_bat_MW",
            "P_sol_sell_MW", "z_grid", "z_solbat", "z_dis", "cycle", "E_MWh", "profit_step",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_output_row_count_equals_n_sim_days_times_n_d(self):
        df = self._minimal_df()
        start_dt = df["timestamp"].iloc[0].to_pydatetime()
        with patch.object(bt, "run_single_optimize", side_effect=_idle_optimizer):
            result = run_backtest(df, start_dt, n_sim_days=1, commit_periods=16, solver=None)
        assert len(result) == N_D

    def test_commit_periods_16_triggers_6_optimizer_calls_per_day(self):
        df = self._minimal_df()
        start_dt = df["timestamp"].iloc[0].to_pydatetime()
        call_count = 0

        def counting_optimizer(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return None, {"solver_status": "ok", "termination_condition": "feasible", "error": ""}

        with patch.object(bt, "run_single_optimize", side_effect=counting_optimizer):
            run_backtest(df, start_dt, n_sim_days=1, commit_periods=16, solver=None)

        assert call_count == 6  # 96 steps / 16 per-commit window

    def test_commit_periods_1_triggers_replan_every_step(self):
        df = self._minimal_df()
        start_dt = df["timestamp"].iloc[0].to_pydatetime()
        call_count = 0

        def counting_optimizer(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return None, {"solver_status": "ok", "termination_condition": "feasible", "error": ""}

        with patch.object(bt, "run_single_optimize", side_effect=counting_optimizer):
            run_backtest(df, start_dt, n_sim_days=1, commit_periods=1, solver=None)

        assert call_count == N_D

    def test_soc_stays_within_bounds(self):
        df = self._minimal_df()
        start_dt = df["timestamp"].iloc[0].to_pydatetime()
        with patch.object(bt, "run_single_optimize", side_effect=_idle_optimizer):
            result = run_backtest(df, start_dt, n_sim_days=1, commit_periods=16, solver=None)
        assert (result["E_MWh"] >= E_MIN - 1e-6).all()
        assert (result["E_MWh"] <= E_MAX + 1e-6).all()

    def test_idle_fallback_yields_zero_profit_per_step(self):
        """p_grid=p_dis=p_sol_sell=0 → profit = price × 0 × DT = 0."""
        df = self._minimal_df()
        start_dt = df["timestamp"].iloc[0].to_pydatetime()
        with patch.object(bt, "run_single_optimize", side_effect=_idle_optimizer):
            result = run_backtest(df, start_dt, n_sim_days=1, commit_periods=16, solver=None)
        assert (result["profit_step"].abs() < 1e-9).all()

    def test_price_source_perfect_does_not_call_prediction(self):
        df = self._minimal_df()
        start_dt = df["timestamp"].iloc[0].to_pydatetime()
        with patch.object(bt, "run_single_optimize", side_effect=_idle_optimizer):
            with patch.object(bt, "build_predicted_price_dict") as mock_pred:
                run_backtest(
                    df, start_dt, n_sim_days=1, commit_periods=16,
                    solver=None, price_source="perfect",
                )
        mock_pred.assert_not_called()

    def test_price_source_predicted_calls_prediction_at_each_replan(self):
        df = self._minimal_df()
        start_dt = df["timestamp"].iloc[0].to_pydatetime()
        dummy_prices = {i: 100.0 for i in range(1, N_D + 1)}
        with patch.object(bt, "run_single_optimize", side_effect=_idle_optimizer):
            with patch.object(
                bt, "build_predicted_price_dict", return_value=dummy_prices
            ) as mock_pred:
                run_backtest(
                    df, start_dt, n_sim_days=1, commit_periods=16,
                    solver=None, price_source="predicted",
                )
        # 6 replans → 6 prediction calls
        assert mock_pred.call_count == 6

    def test_timestamps_in_output_match_price_df(self):
        df = self._minimal_df()
        start_dt = df["timestamp"].iloc[0].to_pydatetime()
        with patch.object(bt, "run_single_optimize", side_effect=_idle_optimizer):
            result = run_backtest(df, start_dt, n_sim_days=1, commit_periods=16, solver=None)
        expected_ts = df["timestamp"].iloc[:N_D]
        pd.testing.assert_series_equal(
            result["timestamp"].reset_index(drop=True),
            expected_ts.reset_index(drop=True),
            check_names=False,
        )

    def test_actual_prices_in_output_come_from_df_all(self):
        df = self._minimal_df()
        df.loc[5, "price"] = 777.0
        start_dt = df["timestamp"].iloc[0].to_pydatetime()
        with patch.object(bt, "run_single_optimize", side_effect=_idle_optimizer):
            result = run_backtest(df, start_dt, n_sim_days=1, commit_periods=16, solver=None)
        assert result["price"].iloc[5] == pytest.approx(777.0)

    def test_trace_records_populated_when_list_passed(self):
        df = self._minimal_df()
        start_dt = df["timestamp"].iloc[0].to_pydatetime()
        trace: list[dict] = []
        with patch.object(bt, "run_single_optimize", side_effect=_idle_optimizer):
            run_backtest(
                df, start_dt, n_sim_days=1, commit_periods=16,
                solver=None, trace_records=trace,
            )
        assert len(trace) == N_D
        assert "step" in trace[0]
        assert "actual_price" in trace[0]

    def test_trace_csv_written_to_disk(self, tmp_path):
        df = self._minimal_df()
        start_dt = df["timestamp"].iloc[0].to_pydatetime()
        trace_path = str(tmp_path / "trace.csv")
        with patch.object(bt, "run_single_optimize", side_effect=_idle_optimizer):
            run_backtest(
                df, start_dt, n_sim_days=1, commit_periods=16,
                solver=None, trace_output_path=trace_path,
            )
        trace_df = pd.read_csv(trace_path)
        assert len(trace_df) == N_D
        assert "step" in trace_df.columns

    def test_stops_early_when_data_runs_out(self):
        # Only 100 rows: first step needs N_D look-ahead (96), but after 4 steps
        # (100 - 96 = 4 safe steps) the guard fires and loop breaks early.
        df = _make_price_df(100)
        start_dt = df["timestamp"].iloc[0].to_pydatetime()
        with patch.object(bt, "run_single_optimize", side_effect=_idle_optimizer):
            result = run_backtest(df, start_dt, n_sim_days=1, commit_periods=16, solver=None)
        assert len(result) < N_D
