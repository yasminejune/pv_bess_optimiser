"""Output formatting utilities for optimization results.

Provides utilities for formatting and presenting optimization results
in user-friendly formats with recommendations and analysis.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def create_recommendations_report(
    results_df: pd.DataFrame, config: Any, output_path: str | None = None
) -> str:
    """Create detailed recommendations report from optimization results.

    Args:
        results_df (pd.DataFrame): Results DataFrame with optimization outputs
        config (Any): Optimization configuration
        output_path (str | None): Optional path to save report file

    Returns:
        str: Formatted recommendations report as string
    """
    report_lines = []

    # Header
    report_lines.extend(
        [
            "BATTERY OPTIMIZATION RECOMMENDATIONS REPORT",
            "=" * 50,
            f"Configuration: {config.config_name}",
            f"Period: {config.optimization.optimization_date} ({config.optimization.duration_hours}h)",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]
    )

    # Executive Summary
    total_profit = results_df["profit_step"].sum()
    total_cycles = results_df["cycle"].sum()
    energy_range = results_df["E_MWh"].max() - results_df["E_MWh"].min()

    report_lines.extend(
        [
            "EXECUTIVE SUMMARY",
            "-" * 20,
            f"Total Profit: £{total_profit:.2f}",
            f"Profit per Hour: £{total_profit / config.optimization.duration_hours:.2f}",
            f"Cycles Used: {total_cycles}/{config.battery.max_cycles_per_day}",
            f"Energy Range: {energy_range:.1f} MWh ({energy_range/config.battery.energy_capacity_mwh*100:.1f}% of capacity)",
            "",
        ]
    )

    # Key Operational Periods
    report_lines.extend(["KEY OPERATIONAL PERIODS", "-" * 25])

    # Charging periods
    charging_mask = (results_df["z_grid"] > 0.5) | (results_df["z_solbat"] > 0.5)
    charging_periods = results_df[charging_mask]

    if len(charging_periods) > 0:
        avg_charge_price = (
            charging_periods["price_intraday"]
            * (charging_periods["P_grid_MW"] + charging_periods["P_sol_bat_MW"])
        ).sum() / (charging_periods["P_grid_MW"] + charging_periods["P_sol_bat_MW"]).sum()
        report_lines.extend(
            [
                "Charging Periods:",
                f"  • {len(charging_periods)} timesteps of charging",
                f"  • Average charging price: £{avg_charge_price:.2f}/MWh",
                f"  • Peak charging power: {(charging_periods['P_grid_MW'] + charging_periods['P_sol_bat_MW']).max():.1f} MW",
            ]
        )

    # Discharging periods
    discharging_mask = results_df["z_dis"] > 0.5
    discharging_periods = results_df[discharging_mask]

    if len(discharging_periods) > 0:
        avg_discharge_price = (
            discharging_periods["price_intraday"] * discharging_periods["P_dis_MW"]
        ).sum() / discharging_periods["P_dis_MW"].sum()
        report_lines.extend(
            [
                "Discharging Periods:",
                f"  • {len(discharging_periods)} timesteps of discharging",
                f"  • Average discharge price: £{avg_discharge_price:.2f}/MWh",
                f"  • Peak discharge power: {discharging_periods['P_dis_MW'].max():.1f} MW",
            ]
        )

    report_lines.append("")

    # Price Analysis
    price_stats = {
        "min": results_df["price_intraday"].min(),
        "max": results_df["price_intraday"].max(),
        "avg": results_df["price_intraday"].mean(),
        "std": results_df["price_intraday"].std(),
    }

    report_lines.extend(
        [
            "PRICE ANALYSIS",
            "-" * 15,
            f"Price range: £{price_stats['min']:.2f} - £{price_stats['max']:.2f}/MWh",
            f"Average price: £{price_stats['avg']:.2f}/MWh",
            f"Price volatility: £{price_stats['std']:.2f}/MWh (std dev)",
            f"Arbitrage spread: £{price_stats['max'] - price_stats['min']:.2f}/MWh",
            "",
        ]
    )

    # Solar Integration (if applicable)
    if config.has_pv:
        solar_to_battery = (
            results_df["P_sol_bat_MW"].sum() * config.optimization.time_step_minutes / 60
        )
        solar_direct = (
            results_df["P_sol_sell_MW"].sum() * config.optimization.time_step_minutes / 60
        )
        total_solar = results_df["solar_MW"].sum() * config.optimization.time_step_minutes / 60

        report_lines.extend(
            [
                "SOLAR INTEGRATION",
                "-" * 17,
                f"Total solar generation: {total_solar:.1f} MWh",
                f"Solar to battery: {solar_to_battery:.1f} MWh ({solar_to_battery/total_solar*100:.1f}%)",
                f"Solar direct export: {solar_direct:.1f} MWh ({solar_direct/total_solar*100:.1f}%)",
                "",
            ]
        )

    # Operational Recommendations
    recommendations = _generate_detailed_recommendations(results_df, config)

    report_lines.extend(["OPERATIONAL RECOMMENDATIONS", "-" * 28])

    for i, rec in enumerate(recommendations, 1):
        report_lines.append(f"{i}. {rec}")

    if not recommendations:
        report_lines.append("No specific recommendations - operation appears optimal.")

    report_lines.append("")

    # Hourly Summary Table
    hourly_summary = create_hourly_summary(results_df, config)
    report_lines.extend(["HOURLY OPERATION SUMMARY", "-" * 25, hourly_summary, ""])

    # Footer
    report_lines.extend(
        [
            "=" * 50,
            "End of Report",
            "",
            "Note: This report is generated automatically based on optimization results.",
            "Please review all recommendations in the context of operational constraints",
            "and market conditions not captured in the optimization model.",
        ]
    )

    report_text = "\n".join(report_lines)

    # Save to file if path provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)

    return report_text


def create_hourly_summary(results_df: pd.DataFrame, config: Any) -> str:
    """Create hourly operation summary table.

    Args:
        results_df (pd.DataFrame): Results DataFrame with optimization outputs
        config (Any): Optimization configuration

    Returns:
        str: Formatted hourly summary table
    """
    # Group by hour
    results_df = results_df.copy()
    results_df["hour"] = results_df.index * config.optimization.time_step_minutes // 60

    hourly_groups = results_df.groupby("hour")

    # Create summary table
    summary_lines = [
        "Hour | Price (£/MWh) | Mode      | Power (MW) | Energy (MWh) | Profit (£)",
        "-----+---------------+-----------+------------+--------------+-----------",
    ]

    for hour, group in hourly_groups:
        avg_price = group["price_intraday"].mean()
        total_profit = group["profit_step"].sum()
        avg_energy = group["E_MWh"].mean()

        # Determine dominant mode
        if (group["z_grid"] > 0.5).any() or (group["z_solbat"] > 0.5).any():
            mode = "Charging"
            power = (group["P_grid_MW"] + group["P_sol_bat_MW"]).mean()
        elif (group["z_dis"] > 0.5).any():
            mode = "Discharge"
            power = -group["P_dis_MW"].mean()
        else:
            mode = "Idle"
            power = 0.0

        summary_lines.append(
            f"{hour:4d} | {avg_price:11.2f} | {mode:9s} | {power:8.1f} | {avg_energy:10.1f} | {total_profit:8.2f}"
        )

    return "\n".join(summary_lines)


def _generate_detailed_recommendations(results_df: pd.DataFrame, config: Any) -> list[str]:
    """Generate detailed operational recommendations.

    Args:
        results_df (pd.DataFrame): Results DataFrame with optimization outputs
        config (Any): Optimization configuration

    Returns:
        list[str]: List of operational recommendation strings
    """
    recommendations = []

    # Cycle utilization analysis
    cycles_used = results_df["cycle"].sum()
    max_cycles = config.battery.max_cycles_per_day

    if cycles_used < max_cycles * 0.8:  # Less than 80% utilization
        recommendations.append(
            f"Consider more aggressive cycling - only {cycles_used}/{max_cycles} cycles used "
            f"({cycles_used/max_cycles*100:.0f}% utilization)"
        )

    # Price arbitrage opportunities
    price_range = results_df["price_intraday"].max() - results_df["price_intraday"].min()
    if price_range > 50:  # High price spread
        charging_at_peak = results_df[
            (results_df["price_intraday"] > results_df["price_intraday"].quantile(0.8))
            & ((results_df["z_grid"] > 0.5) | (results_df["z_solbat"] > 0.5))
        ]

        if len(charging_at_peak) > 0:
            recommendations.append(
                f"Avoid charging during high price periods (£{price_range:.2f}/MWh spread available)"
            )

    # Energy utilization analysis
    energy_range = results_df["E_MWh"].max() - results_df["E_MWh"].min()
    capacity = config.battery.energy_capacity_mwh

    if energy_range < capacity * 0.5:  # Low utilization
        recommendations.append(
            f"Battery capacity underutilized - only {energy_range:.1f} MWh range "
            f"({energy_range/capacity*100:.0f}% of {capacity:.0f} MWh capacity)"
        )

    # Solar integration recommendations
    if config.has_pv:
        solar_total = results_df["solar_MW"].sum()
        solar_to_battery = results_df["P_sol_bat_MW"].sum()

        if solar_total > 0:
            storage_ratio = solar_to_battery / solar_total
            if storage_ratio < 0.3:  # Low solar storage
                recommendations.append(
                    f"Consider storing more solar generation - only {storage_ratio*100:.0f}% stored for later use"
                )
            elif storage_ratio > 0.8:  # High solar storage
                recommendations.append(
                    f"High solar storage rate ({storage_ratio*100:.0f}%) - verify sufficient discharge opportunities"
                )

    # Operational pattern recommendations
    idle_periods = len(
        results_df[
            (results_df["z_grid"] < 0.5)
            & (results_df["z_solbat"] < 0.5)
            & (results_df["z_dis"] < 0.5)
        ]
    )

    if idle_periods > len(results_df) * 0.5:  # More than 50% idle
        recommendations.append(
            f"High idle time ({idle_periods/len(results_df)*100:.0f}% of period) - "
            "consider parameters that encourage more active operation"
        )

    return recommendations


def export_csv_with_metadata(
    results_df: pd.DataFrame, config: Any, output_path: str, include_metadata: bool = True
) -> None:
    """Export results CSV with metadata header comments.

    Args:
        results_df (pd.DataFrame): Results DataFrame
        config (Any): Optimization configuration
        output_path (str): Output CSV file path
        include_metadata (bool): Whether to include metadata header
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Prepare results with user-friendly column names
    export_df = results_df.copy()

    # Add timestamp if not present
    if "timestamp" not in export_df.columns:
        export_df["timestamp"] = pd.date_range(
            start=config.optimization_start_datetime,
            periods=len(export_df),
            freq=f"{config.optimization.time_step_minutes}min",
        )

    # Reorder columns for user-friendliness
    column_order = [
        "timestamp",
        "price_intraday",
        "solar_MW",
        "P_grid_MW",
        "P_dis_MW",
        "P_sol_bat_MW",
        "P_sol_sell_MW",
        "E_MWh",
        "profit_step",
        "z_grid",
        "z_solbat",
        "z_dis",
        "cycle",
    ]

    # Only include columns that exist
    available_columns = [col for col in column_order if col in export_df.columns]
    export_df = export_df[available_columns]

    # Rename columns for clarity
    column_renames = {
        "price_intraday": "Price_GBP_per_MWh",
        "solar_MW": "Solar_Generation_MW",
        "P_grid_MW": "Grid_Charging_MW",
        "P_dis_MW": "Battery_Discharge_MW",
        "P_sol_bat_MW": "Solar_to_Battery_MW",
        "P_sol_sell_MW": "Solar_Direct_Export_MW",
        "E_MWh": "Battery_Energy_MWh",
        "profit_step": "Profit_GBP",
        "z_grid": "Grid_Charge_Mode",
        "z_solbat": "Solar_Charge_Mode",
        "z_dis": "Discharge_Mode",
        "cycle": "Cycle_Event",
    }

    export_df = export_df.rename(columns=column_renames)

    if include_metadata:
        # Create metadata header
        total_profit = export_df["Profit_GBP"].sum()
        total_cycles = export_df["Cycle_Event"].sum()

        metadata_lines = [
            "# Battery Energy Storage Optimization Results",
            f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Configuration: {config.config_name}",
            f"# Period: {config.optimization.optimization_date} ({config.optimization.duration_hours}h)",
            f"# Battery: {config.battery.rated_power_mw}MW / {config.battery.energy_capacity_mwh}MWh",
            f"# Total Profit: £{total_profit:.2f}",
            f"# Cycles Used: {total_cycles}/{config.battery.max_cycles_per_day}",
            "#",
            "# Column Descriptions:",
            "# - timestamp: Date and time for each optimization step",
            "# - Price_GBP_per_MWh: Electricity price (£/MWh)",
            "# - Solar_Generation_MW: Available solar generation (MW)",
            "# - Grid_Charging_MW: Battery charging from grid (MW)",
            "# - Battery_Discharge_MW: Battery discharge to grid (MW)",
            "# - Solar_to_Battery_MW: Solar charging battery directly (MW)",
            "# - Solar_Direct_Export_MW: Solar export to grid directly (MW)",
            "# - Battery_Energy_MWh: Battery stored energy (MWh)",
            "# - Profit_GBP: Profit for this timestep (£)",
            "# - *_Mode columns: Binary indicators (1=active, 0=inactive)",
            "# - Cycle_Event: 1 if new charge cycle starts, 0 otherwise",
            "#",
        ]

        # Write metadata header followed by CSV data
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(metadata_lines) + "\n")
            export_df.to_csv(f, index=False)
    else:
        # Just save CSV without metadata
        export_df.to_csv(output_path, index=False)

    print(f"✓ Results exported to {output_path}")


def create_action_recommendations(results_df: pd.DataFrame) -> pd.DataFrame:
    """Create action recommendations DataFrame for easy interpretation.

    Args:
        results_df (pd.DataFrame): Optimization results DataFrame

    Returns:
        pd.DataFrame: Human-readable action recommendations
    """
    actions_df = results_df[["timestamp"]].copy()

    # Determine recommended action for each timestep
    actions = []
    for _, row in results_df.iterrows():
        if row.get("z_grid", 0) > 0.5:
            action = f"CHARGE from grid at {row['P_grid_MW']:.1f} MW"
        elif row.get("z_solbat", 0) > 0.5:
            action = f"CHARGE from solar at {row['P_sol_bat_MW']:.1f} MW"
        elif row.get("z_dis", 0) > 0.5:
            action = f"DISCHARGE to grid at {row['P_dis_MW']:.1f} MW"
        else:
            action = "IDLE - No battery operation"

        # Add solar export info if applicable
        if row.get("P_sol_sell_MW", 0) > 0.1:
            action += f" + Export {row['P_sol_sell_MW']:.1f} MW solar"

        actions.append(action)

    actions_df["Recommended_Action"] = actions
    actions_df["Price_GBP_per_MWh"] = results_df["price_intraday"]
    actions_df["Battery_SOC_Percent"] = (
        results_df["E_MWh"] / results_df["E_MWh"].max() * 100
    )  # Approximate SOC
    actions_df["Profit_GBP"] = results_df["profit_step"]

    return actions_df
