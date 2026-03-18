"""Example usage of the PV power generation workflow.

This script demonstrates how to use generate_pv_power_for_date_range
to compute 15-minute PV power output using:
  1. A predefined BURST site configuration
  2. A custom PVSiteConfig

It fetches real 15-minute solar irradiance forecast data from the
Open-Meteo API and runs it through the full PV model pipeline.
"""

import pandas as pd

from ors.config.pv_config import PVSiteConfig, SiteType, get_pv_config
from ors.services.weather_to_pv import generate_pv_power_for_date_range
from ors.utils.pv_converter import pv_site_config_to_spec


def run_predefined_site(site_type: SiteType) -> None:
    config = get_pv_config(site_type)
    spec = pv_site_config_to_spec(config)

    print(f"Site:            {config.site_id}")
    print(f"DC Capacity:     {config.pv_capacity_dc_mw:.0f} MW")
    print(f"AC Capacity:     {config.pv_capacity_ac_mw:.0f} MW")
    print(f"Panel Area:      {spec.panel_area_m2:,.0f} m² (derived)")
    print(f"Curtail at:      {config.curtailment_threshold_mw:.0f} MW")
    print()

    results = generate_pv_power_for_date_range(config=config)
    print_results(results)


def run_custom_site() -> None:
    config = PVSiteConfig(
        site_id="Custom_London",
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
    spec = pv_site_config_to_spec(config)

    print(f"Site:            {config.site_id}")
    print(f"DC Capacity:     {config.pv_capacity_dc_mw:.0f} MW")
    print(f"AC Capacity:     {config.pv_capacity_ac_mw:.0f} MW")
    print(f"Panel Area:      {spec.panel_area_m2:,.0f} m² (derived)")
    print(f"Curtail at:      {config.curtailment_threshold_mw:.1f} MW")
    print()

    results = generate_pv_power_for_date_range(config=config)
    print_results(results)


def print_results(df: pd.DataFrame) -> None:
    print(f"{'Timestamp (UTC)':<22} {'Gen (kW)':>10}")
    print("-" * 34)

    for _, row in df.iterrows():
        print(
            f"{row['timestamp_utc'].strftime('%Y-%m-%d %H:%M'):<22} {row['generation_kw']:>10,.1f}"
        )

    print("-" * 34)
    print(f"Total rows:  {len(df)}")
    print()


def main() -> None:
    print("=" * 60)
    print("PV Power Generation — Predefined Site (BURST_1)")
    print("=" * 60)
    run_predefined_site(SiteType.BURST_1)

    print("=" * 60)
    print("PV Power Generation — Predefined Site (BURST_2)")
    print("=" * 60)
    run_predefined_site(SiteType.BURST_2)

    print("=" * 60)
    print("PV Power Generation — Custom Site")
    print("=" * 60)
    run_custom_site()


if __name__ == "__main__":
    main()
