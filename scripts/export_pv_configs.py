"""Generate PV configuration table for documentation.

This script outputs the PV configuration data in various formats
useful for documentation and reporting.
"""

import csv
import json

from ors.config.pv_config import PV_SITE_CONFIGS, SiteType


def generate_csv_table(output_path: str = "pv_site_configs.csv") -> None:
    """Generate CSV table of all PV site configurations.

    Args:
        output_path: Path to output CSV file
    """
    fieldnames = [
        "Site_ID",
        "PV_Block",
        "PV_Capacity_DC_MW",
        "PV_Capacity_AC_MW",
        "DC_AC_Ratio",
        "Module_Efficiency",
        "Inverter_Efficiency",
        "Performance_Ratio",
        "Degradation_Per_Year",
        "Curtailment_Threshold_MW",
        "Clipping_Loss_Factor",
        "Availability",
        "Forced_Outage_Duration_h",
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for site_type in SiteType:
            config = PV_SITE_CONFIGS[site_type]
            writer.writerow(
                {
                    "Site_ID": config.site_id,
                    "PV_Block": config.pv_block,
                    "PV_Capacity_DC_MW": config.pv_capacity_dc_mw,
                    "PV_Capacity_AC_MW": config.pv_capacity_ac_mw,
                    "DC_AC_Ratio": config.dc_ac_ratio,
                    "Module_Efficiency": config.module_efficiency,
                    "Inverter_Efficiency": config.inverter_efficiency,
                    "Performance_Ratio": config.performance_ratio,
                    "Degradation_Per_Year": config.degradation_per_year,
                    "Curtailment_Threshold_MW": config.curtailment_threshold_mw,
                    "Clipping_Loss_Factor": config.clipping_loss_factor,
                    "Availability": config.availability,
                    "Forced_Outage_Duration_h": config.forced_outage_duration_h,
                }
            )

    print(f"✓ CSV table written to: {output_path}")


def generate_json_export(output_path: str = "pv_site_configs.json") -> None:
    """Generate JSON export of all PV site configurations.

    Args:
        output_path: Path to output JSON file
    """
    configs = {}
    for site_type in SiteType:
        config = PV_SITE_CONFIGS[site_type]
        configs[config.site_id] = {
            "pv_block": config.pv_block,
            "pv_capacity_dc_mw": config.pv_capacity_dc_mw,
            "pv_capacity_ac_mw": config.pv_capacity_ac_mw,
            "dc_ac_ratio": config.dc_ac_ratio,
            "module_efficiency": config.module_efficiency,
            "inverter_efficiency": config.inverter_efficiency,
            "performance_ratio": config.performance_ratio,
            "degradation_per_year": config.degradation_per_year,
            "curtailment_threshold_mw": config.curtailment_threshold_mw,
            "clipping_loss_factor": config.clipping_loss_factor,
            "availability": config.availability,
            "forced_outage_duration_h": config.forced_outage_duration_h,
        }

    with open(output_path, "w") as f:
        json.dump(configs, f, indent=2)

    print(f"✓ JSON export written to: {output_path}")


def generate_markdown_table() -> str:
    """Generate markdown table of PV configurations.

    Returns:
        Markdown formatted table
    """
    lines = [
        "| Site ID | DC Cap (MW) | AC Cap (MW) | DC/AC | Module η | Inverter η | PR | Degrad (%/yr) | Curtail (MW) | Clip Loss | Avail |",
        "|---------|-------------|-------------|-------|----------|------------|----|---------------|--------------|-----------|-------|",
    ]

    for site_type in SiteType:
        config = PV_SITE_CONFIGS[site_type]
        lines.append(
            f"| {config.site_id} | {config.pv_capacity_dc_mw:.0f} | "
            f"{config.pv_capacity_ac_mw:.0f} | {config.dc_ac_ratio:.1f} | "
            f"{config.module_efficiency:.2f} | {config.inverter_efficiency:.3f} | "
            f"{config.performance_ratio:.2f} | {config.degradation_per_year:.2f} | "
            f"{config.curtailment_threshold_mw:.0f} | {config.clipping_loss_factor:.2f} | "
            f"{config.availability:.3f} |"
        )

    return "\n".join(lines)


def main() -> None:
    """Main execution function."""
    print("=" * 60)
    print("PV Configuration Export Tool")
    print("=" * 60)

    # Generate CSV
    generate_csv_table()

    # Generate JSON
    generate_json_export()

    # Generate and print Markdown
    print("\n" + "=" * 60)
    print("Markdown Table:")
    print("=" * 60)
    print("\n" + generate_markdown_table())

    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
