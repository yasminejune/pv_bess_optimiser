"""Tests for generate_pv_power_for_date_range."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from ors.config.pv_config import PV_SITE_CONFIGS, PVSiteConfig, SiteType
from ors.services import weather_to_pv as integ


def _solar_df(timestamps: list[str], radiation: list[float]) -> pd.DataFrame:
    """Build a fake solar_radiance_15_mins return value."""
    return pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp(ts) for ts in timestamps],
            "shortwave_radiation": radiation,
        }
    )


def _one_row_solar_df(ts: str, radiation: float) -> pd.DataFrame:
    return _solar_df([ts], [radiation])


class TestGeneratePvPowerForDateRange:
    """Tests for the generate_pv_power_for_date_range function."""

    @pytest.fixture()
    def burst1_config(self) -> PVSiteConfig:
        return PV_SITE_CONFIGS[SiteType.BURST_1]

    def test_returns_dataframe_with_correct_columns(
        self, burst1_config: PVSiteConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            integ.weather_client,
            "solar_radiance_15_mins",
            lambda client, start_datetime=None, end_datetime=None, **kwargs: _one_row_solar_df(
                "2026-03-01T12:00:00Z", 800.0
            ),
        )

        result = integ.generate_pv_power_for_date_range(
            config=burst1_config,
            client=object(),
        )

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["timestamp_utc", "generation_kw"]
        assert len(result) == 1

    def test_zero_irradiance_returns_zero_generation(
        self, burst1_config: PVSiteConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            integ.weather_client,
            "solar_radiance_15_mins",
            lambda client, start_datetime=None, end_datetime=None, **kwargs: _one_row_solar_df(
                "2026-03-01T00:00:00Z", 0.0
            ),
        )

        result = integ.generate_pv_power_for_date_range(
            config=burst1_config,
            client=object(),
        )

        assert result.iloc[0]["generation_kw"] == pytest.approx(0.0)

    def test_energy_estimation_clamped_at_rated_power(
        self, burst1_config: PVSiteConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """At STC (1000 W/m2), raw generation exceeds AC capacity and is clamped."""
        monkeypatch.setattr(
            integ.weather_client,
            "solar_radiance_15_mins",
            lambda client, start_datetime=None, end_datetime=None, **kwargs: _one_row_solar_df(
                "2026-03-01T12:00:00Z", 1000.0
            ),
        )

        result = integ.generate_pv_power_for_date_range(
            config=burst1_config,
            client=object(),
        )

        # Raw generation = panel_area * efficiency * 1.0 kW/m2 = dc_capacity = 65000 kW
        # But rated_power_kw (AC) = 50000, so clamped
        assert result.iloc[0]["generation_kw"] == pytest.approx(50_000.0)

    def test_multi_row_output(
        self, burst1_config: PVSiteConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            integ.weather_client,
            "solar_radiance_15_mins",
            lambda client, start_datetime=None, end_datetime=None, **kwargs: _solar_df(
                [
                    "2026-03-01T10:00:00Z",
                    "2026-03-01T10:15:00Z",
                    "2026-03-01T10:30:00Z",
                ],
                [500.0, 800.0, 600.0],
            ),
        )

        result = integ.generate_pv_power_for_date_range(
            config=burst1_config,
            client=object(),
        )

        assert len(result) == 3

    def test_curtailment_applied_when_exceeding_threshold(
        self, burst1_config: PVSiteConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Generation between curtailment threshold (48 MW) and rated (50 MW) triggers curtailment.

        irr_kw_m2 * panel_area * efficiency = generation_kw
        (irr / 1000) * (65000 / 0.21) * 0.21 = irr * 65
        Need 48000 < irr*65 < 50000 -> 738.5 < irr < 769.2
        """
        monkeypatch.setattr(
            integ.weather_client,
            "solar_radiance_15_mins",
            lambda client, start_datetime=None, end_datetime=None, **kwargs: _one_row_solar_df(
                "2026-03-01T12:00:00Z", 750.0
            ),
        )

        result = integ.generate_pv_power_for_date_range(
            config=burst1_config,
            client=object(),
        )

        gen_kw = result.iloc[0]["generation_kw"]
        # 750 * 65 = 48750 kW, above curtailment threshold of 48000 kW
        assert gen_kw == pytest.approx(48_750.0)

    def test_below_curtailment_generation_passes_through(
        self, burst1_config: PVSiteConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Low irradiance produces generation below curtailment threshold."""
        monkeypatch.setattr(
            integ.weather_client,
            "solar_radiance_15_mins",
            lambda client, start_datetime=None, end_datetime=None, **kwargs: _one_row_solar_df(
                "2026-03-01T12:00:00Z", 100.0
            ),
        )

        result = integ.generate_pv_power_for_date_range(
            config=burst1_config,
            client=object(),
        )

        # 100 * 65 = 6500 kW, well below curtailment threshold
        assert result.iloc[0]["generation_kw"] == pytest.approx(6_500.0)

    def test_creates_client_when_none(
        self, burst1_config: PVSiteConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        made = {"called": False}

        class FakeClient:
            pass

        def fake_make_client() -> FakeClient:
            made["called"] = True
            return FakeClient()

        def fake_solar(client, start_datetime=None, end_datetime=None, **kwargs):
            assert isinstance(client, FakeClient)
            return _one_row_solar_df("2026-03-01T12:00:00Z", 500.0)

        monkeypatch.setattr(integ.weather_client, "make_client", fake_make_client)
        monkeypatch.setattr(integ.weather_client, "solar_radiance_15_mins", fake_solar)

        integ.generate_pv_power_for_date_range(config=burst1_config)

        assert made["called"] is True

    def test_passes_datetimes_to_solar_radiance(
        self, burst1_config: PVSiteConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}

        def fake_solar(client, start_datetime=None, end_datetime=None, **kwargs):
            captured["start_datetime"] = start_datetime
            captured["end_datetime"] = end_datetime
            return _one_row_solar_df("2026-03-01T12:00:00Z", 500.0)

        monkeypatch.setattr(integ.weather_client, "solar_radiance_15_mins", fake_solar)

        start = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
        end = datetime(2026, 3, 1, 18, 0, tzinfo=timezone.utc)

        integ.generate_pv_power_for_date_range(
            config=burst1_config,
            client=object(),
            start_datetime=start,
            end_datetime=end,
        )

        assert captured["start_datetime"] == start
        assert captured["end_datetime"] == end

    def test_defaults_datetimes_to_none(
        self, burst1_config: PVSiteConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no start/end provided, None is passed through to solar_radiance_15_mins."""
        captured: dict[str, object] = {}

        def fake_solar(client, start_datetime=None, end_datetime=None, **kwargs):
            captured["start_datetime"] = start_datetime
            captured["end_datetime"] = end_datetime
            return _one_row_solar_df("2026-03-01T12:00:00Z", 500.0)

        monkeypatch.setattr(integ.weather_client, "solar_radiance_15_mins", fake_solar)

        integ.generate_pv_power_for_date_range(
            config=burst1_config,
            client=object(),
        )

        assert captured["start_datetime"] is None
        assert captured["end_datetime"] is None

    def test_nan_irradiance_returns_zero_generation(
        self, burst1_config: PVSiteConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            integ.weather_client,
            "solar_radiance_15_mins",
            lambda client, start_datetime=None, end_datetime=None, **kwargs: _one_row_solar_df(
                "2026-03-01T12:00:00Z", float("nan")
            ),
        )

        result = integ.generate_pv_power_for_date_range(
            config=burst1_config,
            client=object(),
        )

        assert result.iloc[0]["generation_kw"] == pytest.approx(0.0)

    def test_custom_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A custom PVSiteConfig produces correct generation for its parameters."""
        config = PVSiteConfig(
            site_id="Test",
            pv_block="PV1",
            pv_capacity_dc_mw=10.0,
            pv_capacity_ac_mw=8.0,
            dc_ac_ratio=1.25,
            module_efficiency=0.20,
            inverter_efficiency=0.98,
            performance_ratio=0.80,
            degradation_per_year=0.5,
            curtailment_threshold_mw=5.0,
            clipping_loss_factor=0.02,
            availability=0.99,
            forced_outage_duration_h=2.0,
        )

        monkeypatch.setattr(
            integ.weather_client,
            "solar_radiance_15_mins",
            lambda client, start_datetime=None, end_datetime=None, **kwargs: _one_row_solar_df(
                "2026-03-01T12:00:00Z", 200.0
            ),
        )

        result = integ.generate_pv_power_for_date_range(
            config=config,
            client=object(),
        )

        # 200 W/m2 = 0.2 kW/m2
        # panel_area = 10000 / (1.0 * 0.20) = 50000 m2
        # generation = 0.2 * 50000 * 0.20 = 2000 kW
        # rated_power = 8000 kW, curtailment = 5000 kW → not clamped
        assert result.iloc[0]["generation_kw"] == pytest.approx(2_000.0)
