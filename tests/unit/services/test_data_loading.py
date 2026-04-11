"""Tests for data_loading module."""

import math
from datetime import date, datetime

import pandas as pd
import pytest
from src.ors.config.optimization_config import (
    BatteryConfiguration,
    OptimizationConfig,
    OptimizationConfiguration,
    PVConfiguration,
)
from src.ors.services.data_loading import (
    DataLoadingError,
    _calculate_terminal_price,
    _generate_dummy_prices,
    _generate_dummy_solar,
    _interpolate_hourly_to_timesteps,
    _load_forecasted_prices,
    _load_forecasted_solar,
    _load_historical_prices,
    _load_historical_solar,
    _load_manual_prices,
    _load_manual_solar,
    load_solar_data,
)


def _make_config(
    price_source="forecast",
    pv_source=None,
    hist_price_path=None,
    manual_price_profile=None,
    terminal_method="average",
    terminal_price=None,
):
    pv = None
    if pv_source:
        pv = PVConfiguration(rated_power_kw=100.0, generation_source=pv_source)

    return OptimizationConfig(
        config_name="test",
        battery=BatteryConfiguration(rated_power_mw=100.0, energy_capacity_mwh=600.0),
        optimization=OptimizationConfiguration(
            optimization_date=date(2026, 1, 1),
            price_source=price_source,
            historical_price_path=hist_price_path,
            manual_price_profile=manual_price_profile,
            terminal_price_method=terminal_method,
            terminal_price_gbp_per_mwh=terminal_price,
        ),
        pv=pv,
    )


# ---------------------------------------------------------------------------
# load_solar_data
# ---------------------------------------------------------------------------


class TestLoadSolarData:
    def test_no_pv_returns_zeros(self):
        config = _make_config()
        start = datetime(2026, 1, 1, 0, 0)
        end = datetime(2026, 1, 2, 0, 0)
        result = load_solar_data(config, start, end)
        assert len(result) == 96
        assert all(v == 0.0 for v in result.values())
        assert sorted(result.keys()) == list(range(1, 97))

    def test_historical_solar_missing_path_raises(self):
        config = _make_config(pv_source="historical")
        config.pv.historical_data_path = None
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 2)
        with pytest.raises(DataLoadingError, match="Historical solar data path"):
            load_solar_data(config, start, end)

    def test_historical_solar_missing_file_raises(self):
        config = _make_config(pv_source="historical")
        config.pv.historical_data_path = "/nonexistent/solar.csv"
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 2)
        with pytest.raises(DataLoadingError):
            load_solar_data(config, start, end)

    def test_manual_solar_missing_profile_raises(self):
        config = _make_config(pv_source="manual")
        config.pv.manual_generation_profile = None
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 2)
        with pytest.raises(DataLoadingError, match="Manual generation profile"):
            load_solar_data(config, start, end)

    def test_forecast_solar_requires_panel_area(self):
        config = _make_config(pv_source="forecast")
        config.pv.location_lat = 51.5
        config.pv.location_lon = -0.12
        config.pv.panel_area_m2 = None
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 2)
        with pytest.raises(DataLoadingError, match="panel_area_m2"):
            load_solar_data(config, start, end)

    def test_forecast_solar_uses_runtime_pv_adapter(self, monkeypatch):
        config = _make_config(pv_source="forecast")
        config.pv.location_lat = 51.5
        config.pv.location_lon = -0.12
        config.pv.panel_area_m2 = 2000.0
        config.pv.max_export_kw = 90.0
        config.pv.min_generation_kw = 5.0
        config.pv.curtailment_supported = False
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 1, 1, 0)
        captured: dict[str, object] = {}

        def fake_get_pv_forecast(**kwargs):
            captured.update(kwargs)
            return [1000.0, 2000.0, 3000.0, 4000.0]

        monkeypatch.setattr(
            "src.ors.services.weather_to_pv.get_pv_forecast",
            fake_get_pv_forecast,
        )

        solar = _load_forecasted_solar(config.pv, start, end, 15)

        assert captured["lat"] == pytest.approx(51.5)
        assert captured["lon"] == pytest.approx(-0.12)
        assert captured["panel_area_m2"] == pytest.approx(2000.0)
        assert captured["max_export_kw"] == pytest.approx(90.0)
        assert captured["min_generation_kw"] == pytest.approx(5.0)
        assert captured["curtailment_supported"] is False
        assert solar == {1: 1.0, 2: 2.0, 3: 3.0, 4: 4.0}


# ---------------------------------------------------------------------------
# _load_historical_prices
# ---------------------------------------------------------------------------


class TestLoadHistoricalPrices:
    def test_missing_path_raises(self):
        config = _make_config()
        with pytest.raises(DataLoadingError, match="not provided"):
            _load_historical_prices(None, datetime(2026, 1, 1), datetime(2026, 1, 2), 15, config)

    def test_missing_file_raises(self):
        config = _make_config()
        with pytest.raises(DataLoadingError, match="not found"):
            _load_historical_prices(
                "/nonexistent.csv", datetime(2026, 1, 1), datetime(2026, 1, 2), 15, config
            )

    def test_no_data_in_range_raises(self, tmp_path):
        csv = tmp_path / "prices.csv"
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=96, freq="15min"),
                "price": [50.0] * 96,
            }
        )
        df.to_csv(csv, index=False)
        config = _make_config()
        with pytest.raises(DataLoadingError, match="No price data"):
            _load_historical_prices(
                str(csv), datetime(2026, 6, 1), datetime(2026, 6, 2), 15, config
            )

    def test_loads_valid_data(self, tmp_path):
        csv = tmp_path / "prices.csv"
        start = datetime(2026, 1, 1)
        ts = pd.date_range(start, periods=96, freq="15min")
        df = pd.DataFrame({"timestamp": ts, "price": [42.0] * 96})
        df.to_csv(csv, index=False)
        config = _make_config()
        price_dict, term_price = _load_historical_prices(
            str(csv), start, datetime(2026, 1, 2), 15, config
        )
        assert len(price_dict) == 96
        assert all(v == pytest.approx(42.0) for v in price_dict.values())


class TestLoadForecastPrices:
    def test_forecast_prices_use_inference_adapter(self, monkeypatch):
        config = _make_config(price_source="forecast")
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 1, 1, 0)
        captured: dict[str, object] = {}

        def fake_get_price_forecast(start_datetime, end_datetime, time_step_minutes):
            captured["start_datetime"] = start_datetime
            captured["end_datetime"] = end_datetime
            captured["time_step_minutes"] = time_step_minutes
            return [50.0, 55.0, 60.0, 65.0]

        monkeypatch.setattr(
            "src.ors.services.price_inference.get_price_forecast",
            fake_get_price_forecast,
        )

        price_dict, terminal_price = _load_forecasted_prices(start, end, 15, config)

        assert captured["start_datetime"] == start
        assert captured["end_datetime"] == end
        assert captured["time_step_minutes"] == 15
        assert price_dict == {1: 50.0, 2: 55.0, 3: 60.0, 4: 65.0}
        assert terminal_price == pytest.approx(57.5)

    def test_load_forecasted_prices_falls_back_to_dummy_on_import_error(self, monkeypatch):
        config = _make_config(price_source="forecast")
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 1, 1, 0)

        monkeypatch.setattr(
            "src.ors.services.price_inference.get_price_forecast",
            lambda *args, **kwargs: (_ for _ in ()).throw(ImportError("missing")),
        )

        price_dict, terminal_price = _load_forecasted_prices(start, end, 15, config)

        assert len(price_dict) == 4
        assert all(isinstance(v, float) for v in price_dict.values())
        assert math.isfinite(terminal_price)


# ---------------------------------------------------------------------------
# _load_manual_prices
# ---------------------------------------------------------------------------


class TestLoadManualPrices:
    def test_missing_profile_raises(self):
        config = _make_config()
        with pytest.raises(DataLoadingError, match="not provided"):
            _load_manual_prices(None, datetime(2026, 1, 1), datetime(2026, 1, 2), 60, config)

    def test_valid_profile(self):
        profile = {"00:00": 30.0, "06:00": 50.0, "12:00": 80.0, "18:00": 60.0}
        config = _make_config(terminal_method="manual", terminal_price=55.0)
        price_dict, term_price = _load_manual_prices(
            profile, datetime(2026, 1, 1), datetime(2026, 1, 1, 18, 0), 60, config
        )
        assert len(price_dict) > 0
        assert term_price == pytest.approx(55.0)


# ---------------------------------------------------------------------------
# _load_historical_solar
# ---------------------------------------------------------------------------


class TestLoadHistoricalSolar:
    def test_missing_path_raises(self):
        with pytest.raises(DataLoadingError, match="not provided"):
            _load_historical_solar(None, datetime(2026, 1, 1), datetime(2026, 1, 2), 15)

    def test_missing_file_raises(self):
        with pytest.raises(DataLoadingError, match="not found"):
            _load_historical_solar(
                "/nonexistent.csv", datetime(2026, 1, 1), datetime(2026, 1, 2), 15
            )

    def test_valid_data(self, tmp_path):
        csv = tmp_path / "solar.csv"
        start = datetime(2026, 1, 1)
        ts = pd.date_range(start, periods=96, freq="15min")
        df = pd.DataFrame({"timestamp": ts, "generation_kw": [5000.0] * 96})
        df.to_csv(csv, index=False)
        solar_dict = _load_historical_solar(str(csv), start, datetime(2026, 1, 2), 15)
        assert len(solar_dict) == 96
        assert all(v == pytest.approx(5.0) for v in solar_dict.values())

    def test_no_data_in_range_raises(self, tmp_path):
        csv = tmp_path / "solar.csv"
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=10, freq="15min"),
                "generation_kw": [100.0] * 10,
            }
        )
        df.to_csv(csv, index=False)
        with pytest.raises(DataLoadingError, match="No solar data"):
            _load_historical_solar(str(csv), datetime(2026, 6, 1), datetime(2026, 6, 2), 15)


# ---------------------------------------------------------------------------
# _load_manual_solar
# ---------------------------------------------------------------------------


class TestLoadManualSolar:
    def test_missing_profile_raises(self):
        with pytest.raises(DataLoadingError, match="not provided"):
            _load_manual_solar(None, datetime(2026, 1, 1), datetime(2026, 1, 2), 60)

    def test_valid_profile(self):
        profile = {"06:00": 0.0, "12:00": 5000.0, "18:00": 0.0}
        solar_dict = _load_manual_solar(
            profile, datetime(2026, 1, 1), datetime(2026, 1, 1, 18, 0), 60
        )
        assert len(solar_dict) > 0


# ---------------------------------------------------------------------------
# _calculate_terminal_price
# ---------------------------------------------------------------------------


class TestCalculateTerminalPrice:
    def test_manual_method_with_value(self):
        config = _make_config(terminal_method="manual", terminal_price=99.0)
        price = _calculate_terminal_price({1: 50.0, 2: 60.0}, config)
        assert price == pytest.approx(99.0)

    def test_manual_method_without_value_raises(self):
        config = _make_config(terminal_method="manual", terminal_price=None)
        with pytest.raises(DataLoadingError, match="not provided"):
            _calculate_terminal_price({1: 50.0}, config)

    def test_average_from_optimization_period(self):
        config = _make_config(terminal_method="average")
        price = _calculate_terminal_price({1: 40.0, 2: 60.0}, config)
        assert price == pytest.approx(50.0)

    def test_average_from_historical_df(self):
        config = _make_config(terminal_method="average")
        hist_df = pd.DataFrame({"price": [100.0] * 100})
        price = _calculate_terminal_price({1: 50.0}, config, hist_df)
        assert price == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# _generate_dummy_prices / _generate_dummy_solar
# ---------------------------------------------------------------------------


class TestDummyGenerators:
    def test_dummy_prices_96_steps(self):
        config = _make_config()
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 2)
        prices, term = _generate_dummy_prices(start, end, 15, config)
        assert len(prices) == 96
        assert all(v >= 20.0 for v in prices.values())
        assert math.isfinite(term)

    def test_dummy_solar_96_steps(self):
        pv_config = PVConfiguration(rated_power_kw=1000.0, generation_source="forecast")
        start = datetime(2026, 1, 1)
        end = datetime(2026, 1, 2)
        solar = _generate_dummy_solar(pv_config, start, end, 15)
        assert len(solar) == 96
        # Night hours should be zero
        assert solar[1] == pytest.approx(0.0)  # Midnight


# ---------------------------------------------------------------------------
# _interpolate_hourly_to_timesteps
# ---------------------------------------------------------------------------


class TestInterpolation:
    def test_basic_interpolation(self):
        start = datetime(2026, 1, 1, 0, 0)
        end = datetime(2026, 1, 1, 2, 0)
        hourly = [
            (datetime(2026, 1, 1, 0, 0), 10.0),
            (datetime(2026, 1, 1, 1, 0), 20.0),
            (datetime(2026, 1, 1, 2, 0), 30.0),
        ]
        result = _interpolate_hourly_to_timesteps(hourly, start, end, 60)
        assert len(result) == 2  # 0:00 and 1:00 (end exclusive)
        assert result[1] == pytest.approx(10.0)
        assert result[2] == pytest.approx(20.0)
