"""PV system configuration for different site types.

This module contains the experimental configuration values for PV systems across
different BURST site deployments.
"""

from dataclasses import dataclass
from enum import Enum


class SiteType(str, Enum):
    """Site deployment types."""

    BURST_1 = "Site_Burst_1"
    BURST_2 = "Site_Burst_2"


@dataclass(frozen=True)
class PVSiteConfig:
    """PV site-specific configuration parameters.

    All capacity values are in MW, efficiencies in decimal (0-1),
    and degradation as percentage per year.

    Attributes:
        site_id: Unique site identifier
        pv_block: PV block identifier (e.g., "PV1")
        pv_capacity_dc_mw: DC capacity of PV array (MW)
        pv_capacity_ac_mw: AC capacity of PV inverter (MW)
        dc_ac_ratio: DC to AC capacity ratio
        module_efficiency: PV module efficiency (0-1)
        inverter_efficiency: Inverter efficiency (0-1)
        performance_ratio: Overall system performance ratio (0-1)
        degradation_per_year: Annual degradation rate (% as decimal, e.g., 0.5 for 0.5%)
        curtailment_threshold_mw: Power curtailment threshold (MW)
        clipping_loss_factor: Clipping loss factor (0-1, e.g., 0.03 for 3%)
        availability: System availability factor (0-1)
        forced_outage_duration_h: Expected forced outage duration (hours)

    """

    site_id: str
    pv_block: str
    pv_capacity_dc_mw: float
    pv_capacity_ac_mw: float
    dc_ac_ratio: float
    module_efficiency: float
    inverter_efficiency: float
    performance_ratio: float
    degradation_per_year: float
    curtailment_threshold_mw: float
    clipping_loss_factor: float
    availability: float
    forced_outage_duration_h: float

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.pv_capacity_dc_mw <= 0:
            raise ValueError(f"pv_capacity_dc_mw must be positive: {self.pv_capacity_dc_mw}")
        if self.pv_capacity_ac_mw <= 0:
            raise ValueError(f"pv_capacity_ac_mw must be positive: {self.pv_capacity_ac_mw}")
        if self.dc_ac_ratio <= 0:
            raise ValueError(f"dc_ac_ratio must be positive: {self.dc_ac_ratio}")
        if not (0 <= self.module_efficiency <= 1):
            raise ValueError(f"module_efficiency must be in [0, 1]: {self.module_efficiency}")
        if not (0 <= self.inverter_efficiency <= 1):
            raise ValueError(f"inverter_efficiency must be in [0, 1]: {self.inverter_efficiency}")
        if not (0 <= self.performance_ratio <= 1):
            raise ValueError(f"performance_ratio must be in [0, 1]: {self.performance_ratio}")
        if self.degradation_per_year < 0:
            raise ValueError(
                f"degradation_per_year must be non-negative: {self.degradation_per_year}"
            )
        if self.curtailment_threshold_mw < 0:
            raise ValueError(
                f"curtailment_threshold_mw must be non-negative: {self.curtailment_threshold_mw}"
            )
        if not (0 <= self.clipping_loss_factor <= 1):
            raise ValueError(f"clipping_loss_factor must be in [0, 1]: {self.clipping_loss_factor}")
        if not (0 <= self.availability <= 1):
            raise ValueError(f"availability must be in [0, 1]: {self.availability}")
        if self.forced_outage_duration_h < 0:
            raise ValueError(
                f"forced_outage_duration_h must be non-negative: {self.forced_outage_duration_h}"
            )


# Experimental configuration values for PV sites
# Based on SNET specifications for BURST deployment scenarios; refer to pv_site_configs.csv for source data.
# Can remain as static values here; can create new instances of PVSiteConfig if different configs are needed.
PV_SITE_CONFIGS = {
    SiteType.BURST_1: PVSiteConfig(
        site_id="Site_Burst_1",
        pv_block="PV1",
        pv_capacity_dc_mw=65.0,
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
    ),
    SiteType.BURST_2: PVSiteConfig(
        site_id="Site_Burst_2",
        pv_block="PV1",
        pv_capacity_dc_mw=130.0,
        pv_capacity_ac_mw=100.0,
        dc_ac_ratio=1.3,
        module_efficiency=0.21,
        inverter_efficiency=0.985,
        performance_ratio=0.82,
        degradation_per_year=0.5,
        curtailment_threshold_mw=95.0,
        clipping_loss_factor=0.03,
        availability=0.995,
        forced_outage_duration_h=1.0,
    ),
}


def get_pv_config(site_type: SiteType | str) -> PVSiteConfig:
    """Get PV configuration for a specific site type.

    Args:
        site_type: Site type identifier (enum or string)

    Returns:
        PV site configuration

    Raises:
        KeyError: If site type is not found
        ValueError: If site_type string doesn't match any known site

    """
    if not isinstance(site_type, SiteType):
        try:
            site_type = SiteType(site_type)
        except ValueError as e:
            raise ValueError(
                f"Unknown site type: {site_type}. " f"Valid options: {[t.value for t in SiteType]}"
            ) from e

    return PV_SITE_CONFIGS[site_type]


def list_available_sites() -> list[str]:
    """List all available site configurations.

    Returns:
        List of site identifiers

    """
    return [site.value for site in SiteType]
