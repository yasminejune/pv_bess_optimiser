"""Tests for output_formatting module."""

from types import SimpleNamespace

import pandas as pd
from src.ors.utils.output_formatting import (
    create_action_recommendations,
    create_hourly_summary,
    create_recommendations_report,
    export_csv_with_metadata,
)


def _make_results_df(n=96, price=50.0, discharge_period=None):
    """Create a synthetic results DataFrame."""
    data = {
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="15min"),
        "price_intraday": [price] * n,
        "solar_MW": [0.0] * n,
        "P_grid_MW": [0.0] * n,
        "P_dis_MW": [0.0] * n,
        "P_sol_bat_MW": [0.0] * n,
        "P_sol_sell_MW": [0.0] * n,
        "E_MWh": [300.0] * n,
        "z_grid": [0] * n,
        "z_solbat": [0] * n,
        "z_dis": [0] * n,
        "cycle": [0] * n,
        "profit_step": [0.0] * n,
    }
    df = pd.DataFrame(data)
    if discharge_period:
        start, end = discharge_period
        df.loc[start:end, "z_dis"] = 1
        df.loc[start:end, "P_dis_MW"] = 50.0
        df.loc[start:end, "profit_step"] = 50.0 * price * 0.25
    return df


def _make_config(has_pv=False):
    """Create a minimal config-like object."""
    battery = SimpleNamespace(
        max_cycles_per_day=3,
        energy_capacity_mwh=600.0,
        rated_power_mw=100.0,
    )
    optimization = SimpleNamespace(
        optimization_date="2026-01-01",
        duration_hours=24,
        time_step_minutes=15,
    )
    output = SimpleNamespace()
    cfg = SimpleNamespace(
        config_name="test-config",
        battery=battery,
        optimization=optimization,
        output=output,
        has_pv=has_pv,
        optimization_start_datetime=pd.Timestamp("2026-01-01"),
    )
    return cfg


class TestCreateRecommendationsReport:
    def test_returns_string(self):
        df = _make_results_df()
        config = _make_config()
        report = create_recommendations_report(df, config)
        assert isinstance(report, str)
        assert "BATTERY OPTIMIZATION RECOMMENDATIONS REPORT" in report

    def test_includes_executive_summary(self):
        df = _make_results_df()
        config = _make_config()
        report = create_recommendations_report(df, config)
        assert "EXECUTIVE SUMMARY" in report
        assert "Total Profit" in report

    def test_includes_price_analysis(self):
        df = _make_results_df()
        config = _make_config()
        report = create_recommendations_report(df, config)
        assert "PRICE ANALYSIS" in report

    def test_includes_charging_periods(self):
        df = _make_results_df()
        df.loc[0:3, "z_grid"] = 1
        df.loc[0:3, "P_grid_MW"] = 80.0
        config = _make_config()
        report = create_recommendations_report(df, config)
        assert "Charging Periods" in report

    def test_includes_discharging_periods(self):
        df = _make_results_df(discharge_period=(48, 55))
        config = _make_config()
        report = create_recommendations_report(df, config)
        assert "Discharging Periods" in report

    def test_saves_to_file(self, tmp_path):
        df = _make_results_df()
        config = _make_config()
        out_path = str(tmp_path / "report.txt")
        report = create_recommendations_report(df, config, output_path=out_path)
        assert (tmp_path / "report.txt").exists()
        assert len(report) > 0

    def test_solar_section_when_pv(self):
        df = _make_results_df()
        df["solar_MW"] = [5.0] * len(df)
        df["P_sol_bat_MW"] = [2.0] * len(df)
        df["P_sol_sell_MW"] = [3.0] * len(df)
        config = _make_config(has_pv=True)
        report = create_recommendations_report(df, config)
        assert "SOLAR INTEGRATION" in report


class TestCreateHourlySummary:
    def test_returns_string_with_header(self):
        df = _make_results_df()
        config = _make_config()
        summary = create_hourly_summary(df, config)
        assert isinstance(summary, str)
        assert "Hour" in summary
        assert "Price" in summary

    def test_detects_charging_mode(self):
        df = _make_results_df()
        df.loc[0:3, "z_grid"] = 1
        df.loc[0:3, "P_grid_MW"] = 50.0
        config = _make_config()
        summary = create_hourly_summary(df, config)
        assert "Charging" in summary

    def test_detects_discharge_mode(self):
        df = _make_results_df(discharge_period=(48, 55))
        config = _make_config()
        summary = create_hourly_summary(df, config)
        assert "Discharge" in summary

    def test_detects_idle_mode(self):
        df = _make_results_df()
        config = _make_config()
        summary = create_hourly_summary(df, config)
        assert "Idle" in summary


class TestExportCsvWithMetadata:
    def test_exports_with_metadata(self, tmp_path):
        df = _make_results_df()
        config = _make_config()
        out_path = str(tmp_path / "results.csv")
        export_csv_with_metadata(df, config, out_path)
        content = (tmp_path / "results.csv").read_text()
        assert content.startswith("#")
        assert "Battery Energy Storage" in content

    def test_exports_without_metadata(self, tmp_path):
        df = _make_results_df()
        config = _make_config()
        out_path = str(tmp_path / "results.csv")
        export_csv_with_metadata(df, config, out_path, include_metadata=False)
        content = (tmp_path / "results.csv").read_text()
        assert not content.startswith("#")

    def test_adds_timestamp_if_missing(self, tmp_path):
        df = _make_results_df()
        df = df.drop(columns=["timestamp"])
        config = _make_config()
        out_path = str(tmp_path / "results.csv")
        export_csv_with_metadata(df, config, out_path, include_metadata=False)
        assert (tmp_path / "results.csv").exists()


class TestCreateActionRecommendations:
    def test_returns_dataframe(self):
        df = _make_results_df()
        result = create_action_recommendations(df)
        assert isinstance(result, pd.DataFrame)
        assert "Recommended_Action" in result.columns

    def test_idle_action(self):
        df = _make_results_df()
        result = create_action_recommendations(df)
        assert all("IDLE" in a for a in result["Recommended_Action"])

    def test_charge_from_grid(self):
        df = _make_results_df()
        df.loc[0, "z_grid"] = 1
        df.loc[0, "P_grid_MW"] = 50.0
        result = create_action_recommendations(df)
        assert "CHARGE from grid" in result.loc[0, "Recommended_Action"]

    def test_charge_from_solar(self):
        df = _make_results_df()
        df.loc[0, "z_solbat"] = 1
        df.loc[0, "P_sol_bat_MW"] = 30.0
        result = create_action_recommendations(df)
        assert "CHARGE from solar" in result.loc[0, "Recommended_Action"]

    def test_discharge(self):
        df = _make_results_df()
        df.loc[0, "z_dis"] = 1
        df.loc[0, "P_dis_MW"] = 80.0
        result = create_action_recommendations(df)
        assert "DISCHARGE" in result.loc[0, "Recommended_Action"]

    def test_solar_export_annotation(self):
        df = _make_results_df()
        df.loc[0, "P_sol_sell_MW"] = 5.0
        result = create_action_recommendations(df)
        assert "Export" in result.loc[0, "Recommended_Action"]
