"""Tests for pv_converter utility."""

from __future__ import annotations

import pytest

from ors.config.pv_config import PV_SITE_CONFIGS, PVSiteConfig, SiteType
from ors.utils.pv_converter import STC_IRRADIANCE_KW_M2, pv_site_config_to_spec


class TestPvSiteConfigToSpec:
    """Tests for pv_site_config_to_spec conversion."""

    @pytest.fixture()
    def burst1_config(self) -> PVSiteConfig:
        return PV_SITE_CONFIGS[SiteType.BURST_1]

    @pytest.fixture()
    def burst2_config(self) -> PVSiteConfig:
        return PV_SITE_CONFIGS[SiteType.BURST_2]

    def test_converts_mw_to_kw(self, burst1_config: PVSiteConfig) -> None:
        spec = pv_site_config_to_spec(burst1_config)

        assert spec.rated_power_kw == pytest.approx(50_000.0)
        assert spec.max_export_kw == pytest.approx(48_000.0)
        assert spec.dc_capacity_kw == pytest.approx(65_000.0)
        assert spec.ac_capacity_kw == pytest.approx(50_000.0)

    def test_derives_panel_area_from_dc_capacity(self, burst1_config: PVSiteConfig) -> None:
        spec = pv_site_config_to_spec(burst1_config)

        expected_area = 65_000.0 / (STC_IRRADIANCE_KW_M2 * 0.21)
        assert spec.panel_area_m2 is not None
        assert spec.panel_area_m2 == pytest.approx(expected_area)

    def test_derives_panel_area_burst2(self, burst2_config: PVSiteConfig) -> None:
        spec = pv_site_config_to_spec(burst2_config)

        expected_area = 130_000.0 / (STC_IRRADIANCE_KW_M2 * 0.21)
        assert spec.panel_area_m2 is not None
        assert spec.panel_area_m2 == pytest.approx(expected_area)

    def test_sets_panel_efficiency(self, burst1_config: PVSiteConfig) -> None:
        spec = pv_site_config_to_spec(burst1_config)
        assert spec.panel_efficiency == pytest.approx(0.21)

    def test_sets_curtailment_supported(self, burst1_config: PVSiteConfig) -> None:
        spec = pv_site_config_to_spec(burst1_config)
        assert spec.curtailment_supported is True

    def test_sets_min_generation_to_zero(self, burst1_config: PVSiteConfig) -> None:
        spec = pv_site_config_to_spec(burst1_config)
        assert spec.min_generation_kw == pytest.approx(0.0)

    def test_passes_through_optional_fields(self, burst1_config: PVSiteConfig) -> None:
        spec = pv_site_config_to_spec(burst1_config)

        assert spec.dc_ac_ratio == pytest.approx(1.3)
        assert spec.inverter_efficiency == pytest.approx(0.985)
        assert spec.performance_ratio == pytest.approx(0.82)
        assert spec.degradation_per_year == pytest.approx(0.5)
        assert spec.clipping_loss_factor == pytest.approx(0.03)
        assert spec.availability == pytest.approx(0.995)
        assert spec.forced_outage_duration_h == pytest.approx(1.0)

    def test_stc_irradiance_constant(self) -> None:
        assert pytest.approx(1.0) == STC_IRRADIANCE_KW_M2
