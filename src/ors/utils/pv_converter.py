"""Utilities for converting PV configurations to domain models."""

from ors.config.pv_config import PVSiteConfig
from ors.domain.models.pv import PVSpec

# Standard Test Conditions irradiance (kW/m^2).
# Used to derive panel area from DC capacity and module efficiency.
STC_IRRADIANCE_KW_M2 = 1.0


def pv_site_config_to_spec(config: PVSiteConfig) -> PVSpec:
    """Convert PVSiteConfig to PVSpec domain model.

    Converts MW-based site configuration into kW-based domain specification.
    Panel area is derived from DC capacity and module efficiency at Standard
    Test Conditions (1.0 kW/m²): ``area = dc_kw / (STC * efficiency)``.

    Args:
        config: Site-specific PV configuration

    Returns:
        PV specification compatible with domain models

    """
    # Convert MW to kW
    rated_power_kw = config.pv_capacity_ac_mw * 1000.0
    max_export_kw = config.curtailment_threshold_mw * 1000.0
    dc_capacity_kw = config.pv_capacity_dc_mw * 1000.0
    ac_capacity_kw = config.pv_capacity_ac_mw * 1000.0

    # Derive panel area from DC capacity and module efficiency at STC
    panel_area_m2 = dc_capacity_kw / (STC_IRRADIANCE_KW_M2 * config.module_efficiency)

    return PVSpec(
        rated_power_kw=rated_power_kw,
        max_export_kw=max_export_kw,
        min_generation_kw=0.0,
        curtailment_supported=True,
        panel_area_m2=panel_area_m2,
        panel_efficiency=config.module_efficiency,
        dc_capacity_kw=dc_capacity_kw,
        ac_capacity_kw=ac_capacity_kw,
        dc_ac_ratio=config.dc_ac_ratio,
        inverter_efficiency=config.inverter_efficiency,
        performance_ratio=config.performance_ratio,
        degradation_per_year=config.degradation_per_year,
        clipping_loss_factor=config.clipping_loss_factor,
        availability=config.availability,
        forced_outage_duration_h=config.forced_outage_duration_h,
    )
