"""Tests for PV configuration and conversion."""

import pytest

from ors.config.pv_config import PVSiteConfig, SiteType, get_pv_config, list_available_sites
from ors.utils.pv_converter import pv_site_config_to_spec


class TestPVSiteConfig:
    """Test PV site configuration."""

    def test_burst_1_config(self):
        """Test Site_Burst_1 configuration values."""
        config = get_pv_config(SiteType.BURST_1)
        assert config.site_id == "Site_Burst_1"
        assert config.pv_capacity_dc_mw == 65.0
        assert config.pv_capacity_ac_mw == 50.0
        assert config.dc_ac_ratio == 1.3
        assert config.module_efficiency == 0.21
        assert config.inverter_efficiency == 0.985
        assert config.performance_ratio == 0.82
        assert config.degradation_per_year == 0.5
        assert config.curtailment_threshold_mw == 48.0
        assert config.clipping_loss_factor == 0.03
        assert config.availability == 0.995

    def test_burst_2_config(self):
        """Test Site_Burst_2 configuration values."""
        config = get_pv_config(SiteType.BURST_2)
        assert config.site_id == "Site_Burst_2"
        assert config.pv_capacity_dc_mw == 130.0
        assert config.pv_capacity_ac_mw == 100.0
        assert config.curtailment_threshold_mw == 95.0

    def test_get_config_by_string(self):
        """Test getting config by string identifier."""
        config = get_pv_config("Site_Burst_1")
        assert config.site_id == "Site_Burst_1"

    def test_invalid_site_type_raises_error(self):
        """Test that invalid site type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown site type"):
            get_pv_config("Invalid_Site")

    def test_list_available_sites(self):
        """Test listing all available site configurations."""
        sites = list_available_sites()
        assert len(sites) == 2
        assert "Site_Burst_1" in sites
        assert "Site_Burst_2" in sites


class TestPVConverter:
    """Test PV configuration to domain model conversion."""

    def test_conversion_to_pvspec(self):
        """Test conversion from PVSiteConfig to PVSpec."""
        config = get_pv_config(SiteType.BURST_1)
        spec = pv_site_config_to_spec(config)

        # Check MW to kW conversion
        assert spec.rated_power_kw == 50000.0  # 50 MW AC
        assert spec.max_export_kw == 48000.0  # 48 MW curtailment threshold
        assert spec.dc_capacity_kw == 65000.0  # 65 MW DC
        assert spec.ac_capacity_kw == 50000.0  # 50 MW AC

        # Check efficiency and ratio parameters
        assert spec.dc_ac_ratio == 1.3
        assert spec.panel_efficiency == 0.21
        assert spec.inverter_efficiency == 0.985
        assert spec.performance_ratio == 0.82
        assert spec.degradation_per_year == 0.5
        assert spec.clipping_loss_factor == 0.03
        assert spec.availability == 0.995
        assert spec.forced_outage_duration_h == 1.0

        # Check defaults
        assert spec.min_generation_kw == 0.0
        assert spec.curtailment_supported is True

    def test_all_sites_can_convert(self):
        """Test that all site configurations can be converted to PVSpec."""
        for site_type in SiteType:
            config = get_pv_config(site_type)
            spec = pv_site_config_to_spec(config)
            assert spec.rated_power_kw > 0
            assert spec.max_export_kw is not None


class TestPVSiteConfigValidation:
    """Test PV site configuration validation."""

    def test_invalid_dc_capacity_raises_error(self):
        """Test that invalid DC capacity raises ValueError."""
        with pytest.raises(ValueError, match="pv_capacity_dc_mw must be positive"):
            PVSiteConfig(
                site_id="Test",
                pv_block="PV1",
                pv_capacity_dc_mw=-10.0,
                pv_capacity_ac_mw=50.0,
                dc_ac_ratio=1.3,
                module_efficiency=0.21,
                inverter_efficiency=0.985,
                performance_ratio=0.82,
                degradation_per_year=0.5,
                curtailment_threshold_mw=48.0,
                clipping_loss_factor=0.03,
                availability=0.995,
                forced_outage_duration_h=1.0,
            )

    def test_invalid_efficiency_raises_error(self):
        """Test that invalid efficiency raises ValueError."""
        with pytest.raises(ValueError, match="module_efficiency must be in"):
            PVSiteConfig(
                site_id="Test",
                pv_block="PV1",
                pv_capacity_dc_mw=65.0,
                pv_capacity_ac_mw=50.0,
                dc_ac_ratio=1.3,
                module_efficiency=1.5,  # Invalid: > 1
                inverter_efficiency=0.985,
                performance_ratio=0.82,
                degradation_per_year=0.5,
                curtailment_threshold_mw=48.0,
                clipping_loss_factor=0.03,
                availability=0.995,
                forced_outage_duration_h=1.0,
            )
