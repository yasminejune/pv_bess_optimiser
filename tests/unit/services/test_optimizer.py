"""
Unit tests for the optimizer module (BESS MILP model).

Tests are split into four categories:
- TestModelStructure:  verify the built model has the right components (no solver).
- TestConstraintRules: verify constraint bound types and equality properties (no solver).
- TestLoadInputs:      verify loading/validation and p_30 calculation (intraday DF is created internally).
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
    """Low price first half (t=1..48), high price second half (t=49..96)."""
    return {t: (20.0 if t <= 48 else 200.0) for t in range(1, 97)}


@pytest.fixture()
def built_model(flat_price, zero_solar) -> ConcreteModel:
    """Pre-built model used in structure/rule tests."""
    return build_model(flat_price, zero_solar, p_30=50.0, cycles_used_today=0, t_boundary=96)


# ---------------------------------------------------------------------------
# Helpers: synthetic data for create_input_df() + historic csv writer
# ---------------------------------------------------------------------------


def _make_intraday_df(n_rows: int = 96, price: float = 50.0, gen_kw: float = 0.0) -> pd.DataFrame:
    """
    Create the DF that load_inputs() uses via create_input_df().
    Must include: timestamp, price_intraday, generation_kw
    """
    start = pd.Timestamp("2026-01-01 00:00:00")
    ts = pd.date_range(start, periods=n_rows, freq="15min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "price_intraday": [price] * n_rows,
            "generation_kw": [gen_kw] * n_rows,
        }
    )


def _write_historic_csv(path: Path, n_days: int = 35, price: float = 50.0) -> None:
    """Write a historic price CSV with n_days × 96 rows. p_30 becomes deterministic."""
    rows = []
    base = pd.Timestamp("2025-01-01", tz="UTC")
    for d in range(n_days):
        for h in range(24):
            for m in (0, 15, 30, 45):
                ts = base + pd.Timedelta(days=d, hours=h, minutes=m)
                rows.append(f"{price},,{ts.isoformat()}")
    path.write_text("price,demand_istdo,timestamp\n" + "\n".join(rows) + "\n")


def _placeholder_intraday_csv(path: Path) -> None:
    """
    load_inputs() still takes an intraday_csv path by signature.
    But the function now builds the intraday DF internally (create_input_df),
    so this file is just a minimal placeholder.
    """
    path.write_text("timestamp,price_intraday,solar_MW\n")


# ---------------------------------------------------------------------------
# TestModelStructure
# ---------------------------------------------------------------------------


class TestModelStructure:
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
        # Your two versions differ a bit (cycles_cur_day / cycles_next_day / daily_cycles).
        # We enforce the core set strictly, and accept either "daily_cycles" OR the split constraints.
        required = [
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
        ]
        for name in required:
            assert hasattr(built_model, name), f"Missing constraint: {name}"

        # Cycle limits: accept either combined or split-by-day.
        has_daily = hasattr(built_model, "daily_cycles")
        has_split = hasattr(built_model, "cycles_cur_day")
        assert (
            has_daily or has_split
        ), "Missing cycle limit constraint(s): expected daily_cycles or cycles_cur_day"

    def test_objective_is_maximise(self, built_model):
        assert built_model.obj.sense == maximize

    def test_energy_balance_covers_t1(self, built_model):
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
        assert 1 in built_model.energy_balance

    def test_q_hold_skips_t1(self, built_model):
        assert 1 not in built_model.q_hold

    def test_solar_balance_rhs_matches_solar_param(self, flat_price, uniform_solar):
        m = build_model(flat_price, uniform_solar, p_30=50.0, cycles_used_today=0, t_boundary=96)
        for t in m.T:
            assert value(m.S[t]) == pytest.approx(10.0)

    def test_grid_charge_limit_is_inequality(self, built_model):
        for t in built_model.T:
            assert not built_model.grid_charge_limit[t].equality

    def test_discharge_limit_is_inequality(self, built_model):
        for t in built_model.T:
            assert not built_model.discharge_limit[t].equality


# ---------------------------------------------------------------------------
# TestLoadInputs (intraday DF created internally, so we monkeypatch create_input_df)
# ---------------------------------------------------------------------------


class TestLoadInputs:
    def test_rejects_intraday_df_with_wrong_row_count(self, tmp_path, monkeypatch):
        hist_csv = tmp_path / "hist.csv"
        _write_historic_csv(hist_csv)

        bad_df = _make_intraday_df(n_rows=10)

        import src.ors.services.optimizer.optimizer as opt_mod

        monkeypatch.setattr(opt_mod, "create_input_df", lambda *args, **kwargs: bad_df)

        dummy_intraday_csv = tmp_path / "intraday.csv"
        _placeholder_intraday_csv(dummy_intraday_csv)

        with pytest.raises(AssertionError, match="Expected 96 timesteps"):
            load_inputs(str(dummy_intraday_csv), str(hist_csv))

    def test_rejects_historic_csv_with_insufficient_rows(self, tmp_path, monkeypatch):
        good_df = _make_intraday_df()

        import src.ors.services.optimizer.optimizer as opt_mod

        monkeypatch.setattr(opt_mod, "create_input_df", lambda *args, **kwargs: good_df)

        dummy_intraday_csv = tmp_path / "intraday.csv"
        _placeholder_intraday_csv(dummy_intraday_csv)

        hist_csv = tmp_path / "hist.csv"
        hist_csv.write_text(
            "price,demand_istdo,timestamp\n"
            + "\n".join(f"50,,2025-01-01T{h:02d}:00:00+00:00" for h in range(10))
            + "\n"
        )

        with pytest.raises(AssertionError, match="at least 30 days"):
            load_inputs(str(dummy_intraday_csv), str(hist_csv))

    def test_price_dict_is_1_indexed_with_96_keys(self, tmp_path, monkeypatch):
        df = _make_intraday_df(price=50.0, gen_kw=0.0)

        import src.ors.services.optimizer.optimizer as opt_mod

        monkeypatch.setattr(opt_mod, "create_input_df", lambda *args, **kwargs: df)

        hist_csv = tmp_path / "hist.csv"
        _write_historic_csv(hist_csv)

        dummy_intraday_csv = tmp_path / "intraday.csv"
        _placeholder_intraday_csv(dummy_intraday_csv)

        price, *_ = load_inputs(str(dummy_intraday_csv), str(hist_csv))
        assert sorted(price.keys()) == list(range(1, 97))

    def test_solar_dict_is_1_indexed_with_96_keys_and_matches_generation_kw(
        self, tmp_path, monkeypatch
    ):
        df = _make_intraday_df(price=50.0, gen_kw=5.0)

        import src.ors.services.optimizer.optimizer as opt_mod

        monkeypatch.setattr(opt_mod, "create_input_df", lambda *args, **kwargs: df)

        hist_csv = tmp_path / "hist.csv"
        _write_historic_csv(hist_csv)

        dummy_intraday_csv = tmp_path / "intraday.csv"
        _placeholder_intraday_csv(dummy_intraday_csv)

        _, solar, *_ = load_inputs(str(dummy_intraday_csv), str(hist_csv))
        assert sorted(solar.keys()) == list(range(1, 97))
        assert all(v == pytest.approx(5.0) for v in solar.values())

    def test_price_values_match_create_input_df(self, tmp_path, monkeypatch):
        df = _make_intraday_df(price=123.45, gen_kw=0.0)

        import src.ors.services.optimizer.optimizer as opt_mod

        monkeypatch.setattr(opt_mod, "create_input_df", lambda *args, **kwargs: df)

        hist_csv = tmp_path / "hist.csv"
        _write_historic_csv(hist_csv)

        dummy_intraday_csv = tmp_path / "intraday.csv"
        _placeholder_intraday_csv(dummy_intraday_csv)

        price, *_ = load_inputs(str(dummy_intraday_csv), str(hist_csv))
        assert all(v == pytest.approx(123.45) for v in price.values())

    def test_p_30_is_finite_float(self, tmp_path, monkeypatch):
        df = _make_intraday_df()

        import src.ors.services.optimizer.optimizer as opt_mod

        monkeypatch.setattr(opt_mod, "create_input_df", lambda *args, **kwargs: df)

        hist_csv = tmp_path / "hist.csv"
        _write_historic_csv(hist_csv)

        dummy_intraday_csv = tmp_path / "intraday.csv"
        _placeholder_intraday_csv(dummy_intraday_csv)

        _, _, p_30, *_ = load_inputs(str(dummy_intraday_csv), str(hist_csv))
        assert isinstance(p_30, float)
        assert math.isfinite(p_30)

    def test_p_30_equals_known_historic_price(self, tmp_path, monkeypatch):
        df = _make_intraday_df()

        import src.ors.services.optimizer.optimizer as opt_mod

        monkeypatch.setattr(opt_mod, "create_input_df", lambda *args, **kwargs: df)

        hist_csv = tmp_path / "hist.csv"
        _write_historic_csv(hist_csv, price=77.0)

        dummy_intraday_csv = tmp_path / "intraday.csv"
        _placeholder_intraday_csv(dummy_intraday_csv)

        _, _, p_30, *_ = load_inputs(str(dummy_intraday_csv), str(hist_csv))
        assert p_30 == pytest.approx(77.0)

    def test_load_inputs_with_real_test_data(self, monkeypatch):
        # load_inputs now calls create_input_df(**kwargs) internally;
        # stub it with the real test CSV (renamed to match expected columns).
        raw = pd.read_csv(_INTRADAY_CSV)
        intraday_df = raw.rename(columns={"solar_MW": "generation_kw"})

        import src.ors.services.optimizer.optimizer as opt_mod

        monkeypatch.setattr(opt_mod, "create_input_df", lambda *a, **kw: intraday_df)

        price, solar, p_30, *_ = load_inputs(str(_INTRADAY_CSV), str(_HISTORIC_CSV))
        assert len(price) == 96
        assert len(solar) == 96
        assert math.isfinite(p_30)


# ---------------------------------------------------------------------------
# TestSolverBehaviour
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSolverBehaviour:
    @pytest.fixture(scope="class")
    def solver(self):
        return SolverFactory("highs")

    def _solve(self, m: ConcreteModel, solver) -> ConcreteModel:
        solver.solve(m)
        return m

    def test_flat_price_no_grid_charging(self, flat_price, zero_solar, solver):
        m = self._solve(
            build_model(flat_price, zero_solar, p_30=50.0, cycles_used_today=0, t_boundary=96),
            solver,
        )
        total_grid = sum(value(m.P_grid[t]) for t in m.T)
        assert total_grid == pytest.approx(0.0, abs=1e-3)

    def test_arbitrage_charges_in_cheap_period(self, arbitrage_price, zero_solar, solver):
        m = self._solve(
            build_model(
                arbitrage_price, zero_solar, p_30=100.0, cycles_used_today=0, t_boundary=96
            ),
            solver,
        )
        charge_cheap = sum(value(m.P_grid[t]) for t in range(1, 49))
        assert charge_cheap > 0.0

    def test_arbitrage_discharges_in_expensive_period(self, arbitrage_price, zero_solar, solver):
        m = self._solve(
            build_model(
                arbitrage_price, zero_solar, p_30=100.0, cycles_used_today=0, t_boundary=96
            ),
            solver,
        )
        dis_expensive = sum(value(m.P_dis[t]) for t in range(49, 97))
        assert dis_expensive > 0.0

    def test_cycle_limit_not_exceeded(self, arbitrage_price, zero_solar, solver):
        m = self._solve(
            build_model(
                arbitrage_price, zero_solar, p_30=100.0, cycles_used_today=0, t_boundary=96
            ),
            solver,
        )
        total_cycles = sum(int(round(value(m.cycle[t]))) for t in m.T)
        assert total_cycles <= MAX_CYCLES_PER_DAY

    def test_energy_stays_within_bounds(self, arbitrage_price, zero_solar, solver):
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
        m = self._solve(
            build_model(flat_price, zero_solar, p_30=500.0, cycles_used_today=0, t_boundary=96),
            solver,
        )
        total_dis = sum(value(m.P_dis[t]) for t in m.T)
        assert total_dis == pytest.approx(0.0, abs=1e-3)

    def test_solution_with_real_test_data(self, solver, monkeypatch):
        # load_inputs calls create_input_df internally; stub it with the real CSV.
        raw = pd.read_csv(_INTRADAY_CSV)
        intraday_df = raw.rename(columns={"solar_MW": "generation_kw"})
        import src.ors.services.optimizer.optimizer as opt_mod

        monkeypatch.setattr(opt_mod, "create_input_df", lambda *a, **kw: intraday_df)

        price, solar, p_30, *_ = load_inputs(str(_INTRADAY_CSV), str(_HISTORIC_CSV))

        # If create_input_df() produced NaN/inf solar (e.g., missing mapping), sanitize.
        solar = {t: (0.0 if not math.isfinite(v) else float(v)) for t, v in solar.items()}

        m = self._solve(
            build_model(price, solar, p_30, cycles_used_today=0, t_boundary=96),
            solver,
        )

        for t in m.T:
            assert value(m.P_grid[t]) >= -1e-4
            assert value(m.P_dis[t]) >= -1e-4
