"""Unit tests for the optimizer module (BESS MILP model).

Tests are split into four categories:
- TestModelStructure:  verify the built model has the right components (no solver).
- TestConstraintRules: verify constraint bound types and equality properties (no solver).
- TestLoadInputs:      verify CSV loading, validation, and p_30 calculation.
- TestSolverBehaviour: verify economically-correct solutions (requires HiGHS, marked slow).
"""

import math
from pathlib import Path

import pandas as pd
import pytest
from pyomo.environ import ConcreteModel, SolverFactory, maximize, value
from src.ors.services.optimizer.optimizer import (
    E_MAX,
    E_MIN,
    MAX_CYCLES_PER_DAY,
    build_model,
    load_inputs,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Paths to existing test data committed to the repo
_INTRADAY_CSV = Path("tests/data/optimizer/bess_test_data_intraday_15min.csv")
_HISTORIC_CSV = Path("tests/data/prediction/price_data.csv")


@pytest.fixture()
def flat_price() -> dict[int, float]:
    """96-step flat price — no arbitrage opportunity."""
    return dict.fromkeys(range(1, 97), 50.0)


@pytest.fixture()
def zero_solar() -> dict[int, float]:
    return dict.fromkeys(range(1, 97), 0.0)


@pytest.fixture()
def uniform_solar() -> dict[int, float]:
    """Constant 10 MW solar generation at every timestep."""
    return dict.fromkeys(range(1, 97), 10.0)


@pytest.fixture()
def arbitrage_price() -> dict[int, float]:
    """Low price first half (t=1..48), high price second half (t=49..96).

    The margin is large enough that charging then discharging is clearly profitable
    even after round-trip losses.
    """
    return {t: (20.0 if t <= 48 else 200.0) for t in range(1, 97)}


@pytest.fixture()
def built_model(flat_price, zero_solar) -> ConcreteModel:
    """Pre-built model with flat prices and no solar — used in structure/rule tests."""
    return build_model(flat_price, zero_solar, p_30=50.0, cycles_used_today=0, t_boundary=96)


# ---------------------------------------------------------------------------
# Helpers for creating synthetic CSV files
# ---------------------------------------------------------------------------


def _write_intraday_csv(path: Path, price: float = 50.0, solar: float = 0.0) -> None:
    """Write a valid 96-row intraday CSV to *path*."""
    rows = [
        f"2026-01-01 {h:02d}:{m:02d}:00,{price},{solar}" for h in range(24) for m in (0, 15, 30, 45)
    ]
    path.write_text("timestamp,price_intraday,solar_MW\n" + "\n".join(rows) + "\n")


def _write_historic_csv(path: Path, n_days: int = 35, price: float = 50.0) -> None:
    """Write a historic price CSV with *n_days* × 96 rows to *path*.

    All prices are set to *price* so that p_30 is deterministic in tests.
    """
    rows = []
    base = pd.Timestamp("2025-01-01", tz="UTC")
    for d in range(n_days):
        for h in range(24):
            for m in (0, 15, 30, 45):
                ts = base + pd.Timedelta(days=d, hours=h, minutes=m)
                rows.append(f"{price},,{ts.isoformat()}")
    path.write_text("price,demand_istdo,timestamp\n" + "\n".join(rows) + "\n")


# ---------------------------------------------------------------------------
# TestModelStructure
# ---------------------------------------------------------------------------


class TestModelStructure:
    """Verify the Pyomo model has the correct structure without invoking a solver."""

    def test_model_is_concrete_model(self, built_model):
        assert isinstance(built_model, ConcreteModel)

    def test_timestep_count_matches_price_length(self, flat_price, zero_solar):
        price_4 = dict.fromkeys(range(1, 5), 50.0)
        solar_4 = dict.fromkeys(range(1, 5), 0.0)
        m = build_model(price_4, solar_4, p_30=50.0, cycles_used_today=0, t_boundary=4)
        assert list(m.T) == [1, 2, 3, 4]

    def test_model_has_96_timesteps_for_full_day(self, built_model):
        assert len(list(built_model.T)) == 96

    def test_power_variables_defined(self, built_model):
        for var_name in ("P_grid", "P_dis", "P_sol_bat", "P_sol_sell"):
            assert hasattr(built_model, var_name), f"Missing variable: {var_name}"

    def test_binary_mode_variables_defined(self, built_model):
        for var_name in ("z_grid", "z_solbat", "z_dis"):
            assert hasattr(built_model, var_name), f"Missing variable: {var_name}"

    def test_state_and_cycle_variables_defined(self, built_model):
        for var_name in ("E", "q", "s_dis", "cycle"):
            assert hasattr(built_model, var_name), f"Missing variable: {var_name}"

    def test_all_constraints_defined(self, built_model):
        expected = [
            "mode_excl",
            "solar_balance",
            "grid_charge_limit",
            "sol_charge_limit",
            "discharge_limit",
            "energy_bounds",
            "energy_balance",
            "q_init",
            "s_dis_init",
            "cycle_init",
            "q_on",
            "q_off",
            "q_hold",
            "q_limit",
            "s_dis_def",
            "cycle_and1",
            "cycle_and2",
            "cycle_and3",
            "cycles_cur_day",
            "daily_cycles",
        ]
        for name in expected:
            assert hasattr(built_model, name), f"Missing constraint: {name}"

    def test_objective_is_maximise(self, built_model):
        assert built_model.obj.sense == maximize

    def test_energy_balance_covers_t1(self, built_model):
        # After the bug fix, t=1 is included in energy_balance (no longer skipped)
        assert 1 in built_model.energy_balance

    def test_energy_balance_at_t1_is_equality(self, built_model):
        assert built_model.energy_balance[1].equality

    def test_cycle_init_fixes_cycle_flag_to_zero(self, built_model):
        assert built_model.cycle_init.equality
        assert value(built_model.cycle_init.lower) == pytest.approx(0.0)

    def test_q_init_fixes_charge_flag_to_zero(self, built_model):
        assert built_model.q_init.equality
        assert value(built_model.q_init.lower) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TestConstraintRules
# ---------------------------------------------------------------------------


class TestConstraintRules:
    """Verify constraint bound types and equality properties without a solver."""

    def test_mode_exclusivity_is_inequality(self, built_model):
        for t in built_model.T:
            assert not built_model.mode_excl[t].equality

    def test_mode_exclusivity_upper_bound_is_one(self, built_model):
        for t in built_model.T:
            assert value(built_model.mode_excl[t].upper) == pytest.approx(1.0)

    def test_solar_balance_is_equality(self, built_model):
        for t in built_model.T:
            assert built_model.solar_balance[t].equality

    def test_energy_bounds_lower_is_e_min(self, built_model):
        for t in built_model.T:
            assert value(built_model.energy_bounds[t].lower) == pytest.approx(E_MIN)

    def test_energy_bounds_upper_is_e_max(self, built_model):
        for t in built_model.T:
            assert value(built_model.energy_bounds[t].upper) == pytest.approx(E_MAX)

    def test_energy_balance_includes_t1(self, built_model):
        # After the bug fix, t=1 is part of the energy balance (no longer skipped)
        assert 1 in built_model.energy_balance

    def test_q_hold_skips_t1(self, built_model):
        # Constraint.Skip causes Pyomo to omit t=1 from the dict entirely
        assert 1 not in built_model.q_hold

    def test_solar_balance_rhs_matches_solar_param(self, flat_price, uniform_solar):
        """With 10 MW solar, the solar balance RHS equals 10 at every timestep."""
        m = build_model(flat_price, uniform_solar, p_30=50.0, cycles_used_today=0, t_boundary=96)
        for t in m.T:
            # solar_balance: P_sol_bat + P_sol_sell == S[t] = 10
            assert value(m.S[t]) == pytest.approx(10.0)

    def test_grid_charge_limit_is_inequality(self, built_model):
        for t in built_model.T:
            assert not built_model.grid_charge_limit[t].equality

    def test_discharge_limit_is_inequality(self, built_model):
        for t in built_model.T:
            assert not built_model.discharge_limit[t].equality


# ---------------------------------------------------------------------------
# TestLoadInputs
# ---------------------------------------------------------------------------


class TestLoadInputs:
    """Verify CSV loading, validation errors, and p_30 calculation."""

    def test_rejects_intraday_csv_with_wrong_row_count(self, tmp_path):
        short_csv = tmp_path / "short.csv"
        short_csv.write_text("timestamp,price_intraday,solar_MW\n2026-01-01 00:00:00,50,0\n")
        hist_csv = tmp_path / "hist.csv"
        _write_historic_csv(hist_csv)
        with pytest.raises(AssertionError, match="Expected 96 timesteps"):
            load_inputs(str(short_csv), str(hist_csv))

    def test_rejects_historic_csv_with_insufficient_rows(self, tmp_path):
        intraday_csv = tmp_path / "intraday.csv"
        _write_intraday_csv(intraday_csv)
        hist_csv = tmp_path / "hist.csv"
        # Write only 10 rows — far below the 2880-row minimum
        hist_csv.write_text(
            "price,demand_istdo,timestamp\n"
            + "\n".join(f"50,,2025-01-01 {h:02d}:00:00+00:00" for h in range(10))
            + "\n"
        )
        with pytest.raises(AssertionError, match="at least 30 days"):
            load_inputs(str(intraday_csv), str(hist_csv))

    def test_price_dict_is_1_indexed_with_96_keys(self, tmp_path):
        intraday_csv = tmp_path / "intraday.csv"
        _write_intraday_csv(intraday_csv)
        hist_csv = tmp_path / "hist.csv"
        _write_historic_csv(hist_csv)
        price, *_ = load_inputs(str(intraday_csv), str(hist_csv))
        assert sorted(price.keys()) == list(range(1, 97))

    def test_solar_dict_is_1_indexed_with_96_keys(self, tmp_path):
        intraday_csv = tmp_path / "intraday.csv"
        _write_intraday_csv(intraday_csv, solar=5.0)
        hist_csv = tmp_path / "hist.csv"
        _write_historic_csv(hist_csv)
        _, solar, *_ = load_inputs(str(intraday_csv), str(hist_csv))
        assert sorted(solar.keys()) == list(range(1, 97))
        assert all(v == pytest.approx(5.0) for v in solar.values())

    def test_price_values_match_csv(self, tmp_path):
        intraday_csv = tmp_path / "intraday.csv"
        _write_intraday_csv(intraday_csv, price=123.45)
        hist_csv = tmp_path / "hist.csv"
        _write_historic_csv(hist_csv)
        price, *_ = load_inputs(str(intraday_csv), str(hist_csv))
        assert all(v == pytest.approx(123.45) for v in price.values())

    def test_p_30_is_finite_float(self, tmp_path):
        intraday_csv = tmp_path / "intraday.csv"
        _write_intraday_csv(intraday_csv)
        hist_csv = tmp_path / "hist.csv"
        _write_historic_csv(hist_csv)
        _, _, p_30, *_ = load_inputs(str(intraday_csv), str(hist_csv))
        assert isinstance(p_30, float)
        assert math.isfinite(p_30)

    def test_p_30_equals_known_historic_price(self, tmp_path):
        """When all historic prices are 77.0, p_30 must equal 77.0."""
        intraday_csv = tmp_path / "intraday.csv"
        _write_intraday_csv(intraday_csv)
        hist_csv = tmp_path / "hist.csv"
        _write_historic_csv(hist_csv, price=77.0)
        _, _, p_30, *_ = load_inputs(str(intraday_csv), str(hist_csv))
        assert p_30 == pytest.approx(77.0)

    def test_load_inputs_with_real_test_data(self):
        """Smoke test: the committed test data loads without error."""
        price, solar, p_30, *_ = load_inputs(str(_INTRADAY_CSV), str(_HISTORIC_CSV))
        assert len(price) == 96
        assert len(solar) == 96
        assert math.isfinite(p_30)


# ---------------------------------------------------------------------------
# TestSolverBehaviour
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSolverBehaviour:
    """Verify economically-correct solutions by invoking HiGHS.

    Run with: pytest -m slow
    Skip with: pytest -m "not slow"
    """

    @pytest.fixture(scope="class")
    def solver(self):
        return SolverFactory("highs")

    def _solve(self, m: ConcreteModel, solver) -> ConcreteModel:
        solver.solve(m)
        return m

    def test_flat_price_no_grid_charging(self, flat_price, zero_solar, solver):
        """With flat prices, grid charging then discharging is never profitable."""
        m = self._solve(
            build_model(flat_price, zero_solar, p_30=50.0, cycles_used_today=0, t_boundary=96),
            solver,
        )
        total_grid = sum(value(m.P_grid[t]) for t in m.T)
        assert total_grid == pytest.approx(0.0, abs=1e-3)

    def test_arbitrage_charges_in_cheap_period(self, arbitrage_price, zero_solar, solver):
        """With a clear low/high price split, expect grid charging in the cheap half."""
        m = self._solve(
            build_model(
                arbitrage_price, zero_solar, p_30=100.0, cycles_used_today=0, t_boundary=96
            ),
            solver,
        )
        charge_cheap = sum(value(m.P_grid[t]) for t in range(1, 49))
        assert charge_cheap > 0.0

    def test_arbitrage_discharges_in_expensive_period(self, arbitrage_price, zero_solar, solver):
        """With a clear low/high price split, expect discharge in the expensive half."""
        m = self._solve(
            build_model(
                arbitrage_price, zero_solar, p_30=100.0, cycles_used_today=0, t_boundary=96
            ),
            solver,
        )
        dis_expensive = sum(value(m.P_dis[t]) for t in range(49, 97))
        assert dis_expensive > 0.0

    def test_cycle_limit_not_exceeded(self, arbitrage_price, zero_solar, solver):
        """Total daily cycles must not exceed MAX_CYCLES_PER_DAY."""
        m = self._solve(
            build_model(
                arbitrage_price, zero_solar, p_30=100.0, cycles_used_today=0, t_boundary=96
            ),
            solver,
        )
        total_cycles = sum(int(round(value(m.cycle[t]))) for t in m.T)
        assert total_cycles <= MAX_CYCLES_PER_DAY

    def test_energy_stays_within_bounds(self, arbitrage_price, zero_solar, solver):
        """E[t] must remain in [E_MIN, E_MAX] at every timestep."""
        m = self._solve(
            build_model(
                arbitrage_price, zero_solar, p_30=100.0, cycles_used_today=0, t_boundary=96
            ),
            solver,
        )
        for t in m.T:
            e = value(m.E[t])
            assert E_MIN - 1e-3 <= e <= E_MAX + 1e-3, f"Energy out of bounds at t={t}: {e:.2f}"

    def test_mode_exclusivity_holds_in_solution(self, arbitrage_price, zero_solar, solver):
        """At most one of {z_grid, z_solbat, z_dis} may be active at any timestep."""
        m = self._solve(
            build_model(
                arbitrage_price, zero_solar, p_30=100.0, cycles_used_today=0, t_boundary=96
            ),
            solver,
        )
        for t in m.T:
            mode_sum = (
                round(value(m.z_grid[t])) + round(value(m.z_solbat[t])) + round(value(m.z_dis[t]))
            )
            assert mode_sum <= 1, f"Multiple modes active at t={t}"

    def test_solar_balance_holds_in_solution(self, flat_price, uniform_solar, solver):
        """P_sol_bat[t] + P_sol_sell[t] must equal solar[t] at every timestep."""
        m = self._solve(
            build_model(flat_price, uniform_solar, p_30=50.0, cycles_used_today=0, t_boundary=96),
            solver,
        )
        for t in m.T:
            solar_used = value(m.P_sol_bat[t]) + value(m.P_sol_sell[t])
            assert solar_used == pytest.approx(
                value(m.S[t]), abs=1e-4
            ), f"Solar balance violated at t={t}"

    def test_high_terminal_value_prevents_discharge(self, flat_price, zero_solar, solver):
        """When p_30 >> dispatch price, keeping energy is strongly preferred.

        With the energy balance bug fixed (t=1 now included), discharging at
        any timestep costs real battery energy.  Revenue at 50 £/MWh per MWh
        grid-side is worth only 50 × ETA_DIS = 48.5 £/MWh battery-side,
        well below the 500 £/MWh terminal value, so P_dis must stay at zero.
        """
        m = self._solve(
            build_model(flat_price, zero_solar, p_30=500.0, cycles_used_today=0, t_boundary=96),
            solver,
        )
        total_dis = sum(value(m.P_dis[t]) for t in m.T)
        assert total_dis == pytest.approx(0.0, abs=1e-3)

    def test_solution_with_real_test_data(self, solver):
        """Smoke test: the model solves feasibly on committed test data."""
        price, solar, p_30, *_ = load_inputs(str(_INTRADAY_CSV), str(_HISTORIC_CSV))
        m = self._solve(build_model(price, solar, p_30, cycles_used_today=0, t_boundary=96), solver)
        # Basic sanity: all power values are non-negative
        for t in m.T:
            assert value(m.P_grid[t]) >= -1e-4
            assert value(m.P_dis[t]) >= -1e-4
