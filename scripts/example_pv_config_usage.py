"""Example usage of PV site configurations.

This script demonstrates how to use the new PV configuration system
for BURST site types.
"""

from ors.config.pv_config import SiteType, get_pv_config, list_available_sites
from ors.utils.pv_converter import pv_site_config_to_spec


def print_site_summary(site_type: SiteType) -> None:
    """Print a summary of a site configuration.

    Args:
        site_type: Site type to summarize
    """
    config = get_pv_config(site_type)
    spec = pv_site_config_to_spec(config)

    print(f"\n{'=' * 60}")
    print(f"Site: {config.site_id} ({config.pv_block})")
    print(f"{'=' * 60}")
    print(f"DC Capacity:        {config.pv_capacity_dc_mw:.1f} MW")
    print(f"AC Capacity:        {config.pv_capacity_ac_mw:.1f} MW")
    print(f"DC/AC Ratio:        {config.dc_ac_ratio:.2f}")
    print(f"Module Efficiency:  {config.module_efficiency * 100:.1f}%")
    print(f"Inverter Efficiency:{config.inverter_efficiency * 100:.2f}%")
    print(f"Performance Ratio:  {config.performance_ratio * 100:.1f}%")
    print(f"Degradation/Year:   {config.degradation_per_year:.2f}%")
    print(f"Curtail Threshold:  {config.curtailment_threshold_mw:.1f} MW")
    print(f"Clipping Loss:      {config.clipping_loss_factor * 100:.1f}%")
    print(f"Availability:       {config.availability * 100:.2f}%")
    print(f"Outage Duration:    {config.forced_outage_duration_h:.1f} h")
    print("\nPVSpec Conversion:")
    print(f"  Rated Power:      {spec.rated_power_kw:.0f} kW")
    print(f"  Max Export:       {spec.max_export_kw:.0f} kW")
    print(f"  DC Capacity:      {spec.dc_capacity_kw:.0f} kW")
    print(f"  AC Capacity:      {spec.ac_capacity_kw:.0f} kW")


def main() -> None:
    """Main example function."""
    print("=" * 60)
    print("PV Site Configuration Examples")
    print("=" * 60)

    # List all available sites
    print(f"\nAvailable sites: {', '.join(list_available_sites())}")

    # Display configuration for each site type
    for site_type in SiteType:
        print_site_summary(site_type)

    # Example: Compare different BURST configurations
    print(f"\n{'=' * 60}")
    print("BURST Configuration Comparison")
    print(f"{'=' * 60}")

    burst_1 = get_pv_config(SiteType.BURST_1)
    burst_2 = get_pv_config(SiteType.BURST_2)

    print(f"\n{'Metric':<25} {'Burst_1':>15} {'Burst_2':>15}")
    print(f"{'-' * 60}")
    print(
        f"{'DC Capacity (MW)':<25} {burst_1.pv_capacity_dc_mw:>15.0f} {burst_2.pv_capacity_dc_mw:>15.0f}"
    )
    print(
        f"{'AC Capacity (MW)':<25} {burst_1.pv_capacity_ac_mw:>15.0f} {burst_2.pv_capacity_ac_mw:>15.0f}"
    )
    print(f"{'DC/AC Ratio':<25} {burst_1.dc_ac_ratio:>15.2f} {burst_2.dc_ac_ratio:>15.2f}")
    print(
        f"{'Performance Ratio':<25} {burst_1.performance_ratio:>15.2f} "
        f"{burst_2.performance_ratio:>15.2f}"
    )
    print(
        f"{'Degradation (%/yr)':<25} {burst_1.degradation_per_year:>15.2f} "
        f"{burst_2.degradation_per_year:>15.2f}"
    )
    print(
        f"{'Clipping Loss (%)':<25} {burst_1.clipping_loss_factor * 100:>15.1f} "
        f"{burst_2.clipping_loss_factor * 100:>15.1f}"
    )

    print("\nKey Characteristics:")
    print("• Both sites have DC/AC ratio of 1.3 for aggressive oversizing")
    print("• Both sites have performance ratio of 0.82")
    print("• Both sites have degradation of 0.5%/year")
    print("• Both sites have clipping loss of 3%")
    print("• Burst_2 has exactly 2x the capacity of Burst_1")


if __name__ == "__main__":
    main()
