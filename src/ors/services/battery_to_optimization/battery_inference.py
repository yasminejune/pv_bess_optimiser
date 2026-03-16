"""Battery inference module for optimizer integration.

This module provides functions to bridge the battery management module
with the optimization results, including CSV logging and validation.
"""

import csv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

# Import battery management functions with proper absolute imports
from ors.services.battery.battery_management import (
    BatteryParams,
    compute_losses,
    create_log_entry,
    load_battery_params_and_defaults,
    step_energy,
    write_simulation_csv,
)


def load_optimizer_battery_config(
    config_path: str | None = None,
) -> tuple[BatteryParams, dict[str, Any]]:
    """Load battery configuration for optimizer.

    Args:
        config_path (str | None): Optional path to config file. If None, uses default location.

    Returns:
        tuple[BatteryParams, dict[str, Any]]: Tuple of (BatteryParams, simulation_defaults)
    """
    if config_path is None:
        config_path = str(Path(__file__).parent.parent / "battery" / "battery_config.json")

    return load_battery_params_and_defaults(config_path)


def create_optimizer_log_entries(
    df_results: pd.DataFrame,
    params: BatteryParams,
    dt_hours: float = 0.25,
    start_datetime: datetime | None = None,
    initial_energy_mwh: float | None = None,
    verbose: bool = False,
) -> list[dict[str, Any]]:
    """Create detailed log entries from optimizer results with step-by-step battery physics.

    This function creates detailed logs for each optimization timestep, similar to the
    battery demo, with complete loss breakdown and energy accounting.

    Args:
        df_results (pd.DataFrame): DataFrame with optimizer results containing columns:
            timestamp, price_intraday, solar_MW, P_grid_MW, P_dis_MW,
            P_sol_bat_MW, P_sol_sell_MW, E_MWh
        params (BatteryParams): Battery parameters
        dt_hours (float): Time step duration in hours
        start_datetime (datetime | None): Optional start time for timestamping
        initial_energy_mwh (float | None): Initial battery energy state for step 0. If None, uses 50% SOC.
        verbose (bool): Enable verbose step-by-step output

    Returns:
        list[dict[str, Any]]: List of detailed log entries compatible with battery module CSV export
    """
    logs = []

    if verbose:
        print(f"Info: Creating detailed step-by-step battery logs for {len(df_results)} timesteps...")
        print(f"Info: DataFrame shape: {df_results.shape}")
        print(f"Info: DataFrame columns: {list(df_results.columns)}")
    else:
        print(f"Info: Processing {len(df_results)} timesteps...")

    try:
        for step, row in df_results.iterrows():
            if verbose:
                print(f"Info: Processing step {step}...")

            # Extract power values from optimizer results
            p_grid_mw = float(row["P_grid_MW"])
            p_solar_mw = float(row["P_sol_bat_MW"])  # Only solar going to battery
            p_dis_mw = float(row["P_dis_MW"])

            # Energy states
            if step == 0:
                # Use provided initial energy or default to 50% SOC
                e_prev_mwh = (
                    initial_energy_mwh if initial_energy_mwh is not None else params.e_cap_mwh * 0.5
                )
            else:
                e_prev_mwh = float(df_results.iloc[step - 1]["E_MWh"])
            e_next_mwh = float(row["E_MWh"])

            if verbose:
                print(
                    f"Info: Powers: Grid={p_grid_mw:.2f}, Solar={p_solar_mw:.2f}, Discharge={p_dis_mw:.2f}"
                )
                print(f"Info: Energy: {e_prev_mwh:.2f} -> {e_next_mwh:.2f} MWh")

            # Calculate losses for this step using battery physics
            losses = compute_losses(
                e_prev_mwh=e_prev_mwh,
                p_grid_mw=p_grid_mw,
                p_sol_mw=p_solar_mw,
                p_dis_mw=p_dis_mw,
                params=params,
                dt_hours=dt_hours,
            )

            # Calculate timing
            t_hours = step * dt_hours
            timestamp_iso = None
            if start_datetime is not None:
                step_datetime = start_datetime + timedelta(hours=t_hours)
                timestamp_iso = step_datetime.isoformat()
            elif hasattr(row, "timestamp") and not pd.isna(row["timestamp"]):
                timestamp_iso = pd.to_datetime(row["timestamp"]).isoformat()

            # Create standardized log entry using battery management function
            if verbose:
                print("Info: Creating log entry...")
            log_entry = create_log_entry(
                step=step,
                t_hours=t_hours,
                dt_hours=dt_hours,
                p_grid_mw=p_grid_mw,
                p_sol_mw=p_solar_mw,
                p_dis_mw=p_dis_mw,
                e_prev_mwh=e_prev_mwh,
                e_next_mwh=e_next_mwh,
                losses=losses,
                eta_ch=params.eta_ch,
                timestamp_iso=timestamp_iso,
            )

            # Note: Only standard battery/energy fields are included in the CSV
            # The log_entry already contains all required battery module fields

            logs.append(log_entry)
            
            if verbose:
                print(f"Success: Step {step} logged")

            # Print step progress every 24 steps (6 hours at 15-min intervals)
            if (step + 1) % 24 == 0 or step == 0 or step == len(df_results) - 1:
                hour = int(t_hours)
                minute = int((t_hours % 1) * 60)

                # Determine battery mode for display
                battery_mode = "Idle"
                if row.get("z_grid", 0) > 0.5:
                    battery_mode = "Grid→Battery"
                elif row.get("z_solbat", 0) > 0.5:
                    battery_mode = "Solar→Battery"
                elif row.get("z_dis", 0) > 0.5:
                    battery_mode = "Battery→Grid"

                if verbose:
                    print(
                        f"Info: Step {step + 1:3d} ({hour:02d}:{minute:02d}): {battery_mode:12s} | "
                        f"E={e_next_mwh:6.1f} MWh | Loss={losses.total_loss_mwh:.3f} MWh"
                    )

        print(f"Success: Created {len(logs)} detailed step logs")
        return logs

    except Exception as e:
        print(f"Error: Error creating logs: {e}")
        import traceback

        traceback.print_exc()
        return []


def export_optimizer_results(
    df_results: pd.DataFrame,
    csv_path: str,
    params: BatteryParams | None = None,
    dt_hours: float = 0.25,
    start_datetime: datetime | None = None,
    config_path: str | None = None,
    initial_energy_mwh: float | None = None,
) -> None:
    """Export optimizer results using battery module CSV format.

    Args:
        df_results (pd.DataFrame): DataFrame with optimizer results
        csv_path (str): Path where CSV should be written
        params (BatteryParams | None): Battery parameters (loaded from config if None)
        dt_hours (float): Time step duration in hours
        start_datetime (datetime | None): Optional start time for timestamping
        config_path (str | None): Optional path to battery config file
        initial_energy_mwh (float | None): Initial battery energy state. If None, uses 50% SOC.
    """
    if params is None:
        params, _ = load_optimizer_battery_config(config_path)

    logs = create_optimizer_log_entries(
        df_results=df_results,
        params=params,
        dt_hours=dt_hours,
        start_datetime=start_datetime,
        initial_energy_mwh=initial_energy_mwh,
    )

    write_simulation_csv(logs, csv_path)


def validate_optimizer_energy_balance(
    df_results: pd.DataFrame, params: BatteryParams, dt_hours: float = 0.25, tolerance: float = 1e-6
) -> dict[str, Any]:
    """Validate that optimizer energy states are consistent with battery physics.

    Args:
        df_results (pd.DataFrame): DataFrame with optimizer results
        params (BatteryParams): Battery parameters
        dt_hours (float): Time step duration in hours
        tolerance (float): Numerical tolerance for validation

    Returns:
        dict[str, Any]: Dictionary with validation results including errors and summary
    """
    validation_results: dict[str, Any] = {
        "is_valid": True,
        "errors": [],
        "max_error": 0.0,
        "energy_drift": 0.0,
        "summary": {},
    }

    errors = []

    for step, row in df_results.iterrows():
        if step == 0:
            continue  # Skip first row (no previous state)

        # Get energy states
        e_prev = df_results.iloc[step - 1]["E_MWh"]
        e_current = row["E_MWh"]

        # Calculate expected energy using battery physics
        p_grid_mw = row["P_grid_MW"]
        p_solar_mw = row["P_sol_bat_MW"]
        p_dis_mw = row["P_dis_MW"]

        e_expected = step_energy(
            e_prev_mwh=e_prev,
            p_grid_mw=p_grid_mw,
            p_sol_mw=p_solar_mw,
            p_dis_mw=p_dis_mw,
            params=params,
            dt_hours=dt_hours,
            enforce_bounds=True,
        )

        # Calculate error
        error = abs(e_current - e_expected)
        errors.append(error)

        if error > tolerance:
            validation_results["is_valid"] = False
            validation_results["errors"].append(
                {
                    "step": step,
                    "optimizer_energy": e_current,
                    "expected_energy": e_expected,
                    "error": error,
                    "timestamp": getattr(row, "timestamp", f"Step {step}"),
                }
            )

    if errors:
        validation_results["max_error"] = max(errors)
        initial_energy = df_results.iloc[0]["E_MWh"]
        final_energy = df_results.iloc[-1]["E_MWh"]
        validation_results["energy_drift"] = final_energy - initial_energy

    validation_results["summary"] = {
        "total_steps": len(df_results),
        "validated_steps": len(errors),
        "failed_steps": len(validation_results["errors"]),
        "max_error_mwh": validation_results["max_error"],
        "avg_error_mwh": sum(errors) / len(errors) if errors else 0.0,
    }

    return validation_results


def create_enhanced_optimizer_output(
    df_results: pd.DataFrame,
    csv_path: str,
    params: BatteryParams | None = None,
    dt_hours: float = 0.25,
    start_datetime: datetime | None = None,
    validate: bool = True,
    config_path: str | None = None,
    initial_energy_mwh: float | None = None,
    verbose: bool = False,
) -> dict[str, Any]:
    """Create enhanced optimizer output with validation and detailed logging.

    This function processes optimizer results and creates detailed battery logs
    with step-by-step recording similar to the battery demo.

    Args:
        df_results (pd.DataFrame): DataFrame with optimizer results
        csv_path (str): Path where enhanced CSV should be written
        params (BatteryParams | None): Battery parameters (loaded from config if None)
        dt_hours (float): Time step duration in hours
        start_datetime (datetime | None): Optional start time for timestamping
        validate (bool): Whether to validate energy balance
        config_path (str | None): Optional path to battery config file
        initial_energy_mwh (float | None): Initial battery energy state. If None, uses 50% SOC.
        verbose (bool): Enable verbose step-by-step output

    Returns:
        dict[str, Any]: Dictionary with processing results and validation info
    """
    if params is None:
        params, _ = load_optimizer_battery_config(config_path)

    print("Info: Creating enhanced battery output with step-by-step logging...")
    if verbose:
        print(f"Info: Battery: {params.p_rated_mw} MW / {params.e_cap_mwh} MWh")
        print(f"Info: Time step: {dt_hours} hours ({dt_hours*60} minutes)")
        print(f"Info: Output path: {csv_path}")

    # Create detailed logs with step-by-step processing
    logs = create_optimizer_log_entries(
        df_results=df_results,
        params=params,
        dt_hours=dt_hours,
        start_datetime=start_datetime,
        initial_energy_mwh=initial_energy_mwh,
        verbose=verbose,
    )

    print(f"Info: Created {len(logs)} log entries")

    if not logs:
        print("Error: No logs created! Cannot write to CSV.")
        return {"csv_path": csv_path, "num_steps": 0, "error": "No logs created"}

    # Debug: Show first log entry
    if verbose:
        print(f"Info: First log entry keys: {list(logs[0].keys())}")
        print(f"Info: First log entry sample: {dict(list(logs[0].items())[:5])}")

    # Export to CSV using battery module function (standard battery format only)
    print(f"Info: Writing detailed logs to {csv_path}...")
    try:
        # Use the standard battery module CSV writer which only includes battery/energy fields
        write_simulation_csv(logs, csv_path)
        print(f"Success: Successfully wrote {len(logs)} step records to CSV")

        # Verify file was created and has content
        csv_file_path = Path(csv_path)
        if csv_file_path.exists():
            if verbose:
                file_size = csv_file_path.stat().st_size
                print(f"Info: CSV file created: {file_size} bytes")

                # Read back a few lines to verify
                with open(csv_path) as f:
                    lines = f.readlines()
                    print(f"Info: CSV has {len(lines)} lines (including header)")
                    if len(lines) > 1:
                        print(f"Info: Header: {lines[0].strip()}")
                        print(f"Info: First row: {lines[1].strip()}")
        else:
            print(f"Error: CSV file was not created at {csv_path}")

    except Exception as e:
        print(f"Error: Error writing CSV: {e}")
        import traceback

        traceback.print_exc()
        return {"csv_path": csv_path, "num_steps": len(logs), "error": f"CSV write failed: {e}"}

    # Calculate summary statistics
    total_profit = sum(log.get("profit_step", 0) for log in logs)
    total_cycles = sum(log.get("cycle", 0) for log in logs)
    total_losses = sum(log.get("loss_total_mwh", 0) for log in logs)
    energy_range = {
        "min": min(log["energy_after_mwh"] for log in logs),
        "max": max(log["energy_after_mwh"] for log in logs),
    }

    results = {
        "csv_path": csv_path,
        "num_steps": len(logs),
        "total_profit": round(total_profit, 2),
        "total_cycles": int(total_cycles),
        "total_losses_mwh": round(total_losses, 3),
        "energy_range_mwh": energy_range,
        "params_used": repr(params),
    }

    # Validate if requested
    if validate:
        print("Info: Validating energy balance...")
        validation = validate_optimizer_energy_balance(
            df_results=df_results, params=params, dt_hours=dt_hours
        )
        results["validation"] = validation

        if validation["is_valid"]:
            print("Success: Energy balance validation passed")
        else:
            print(
                f"Warning: Energy balance validation failed: {validation['summary']['failed_steps']} errors"
            )

    return results


def write_step_by_step_battery_log(
    optimizer_step: int,
    row: pd.Series,
    params: BatteryParams,
    csv_path: str,
    dt_hours: float = 0.25,
    step_datetime: datetime | None = None,
    append_mode: bool = True,
    initial_energy_mwh: float | None = None,
) -> dict[str, Any]:
    """Write a single optimization step to battery CSV log (for real-time logging).

    This function allows writing individual steps as optimization progresses,
    similar to how the battery demo logs each simulation step.

    Args:
        optimizer_step (int): Current optimization timestep (0-based)
        row (pd.Series): Single row of optimizer results (pandas Series)
        params (BatteryParams): Battery parameters
        csv_path (str): Path to CSV file
        dt_hours (float): Time step duration in hours
        step_datetime (datetime | None): Optional timestamp for this step
        append_mode (bool): If True, append to existing file; if False, create new file
        initial_energy_mwh (float | None): Initial battery energy state. If None, uses 50% SOC.

    Returns:
        dict[str, Any]: Dictionary with step log entry that was written
    """
    # Extract power values from optimizer results
    p_grid_mw = row["P_grid_MW"]
    p_solar_mw = row["P_sol_bat_MW"]
    p_dis_mw = row["P_dis_MW"]

    # Calculate energy states (assuming previous energy is available)
    if optimizer_step == 0:
        # Use provided initial energy or default to 50% SOC
        e_prev_mwh = (
            initial_energy_mwh if initial_energy_mwh is not None else params.e_cap_mwh * 0.5
        )
    else:
        # In real implementation, this would come from previous step
        e_prev_mwh = row.get("E_prev_MWh", params.e_cap_mwh * 0.5)

    e_next_mwh = row["E_MWh"]

    # Calculate losses and create log entry
    losses = compute_losses(
        e_prev_mwh=e_prev_mwh,
        p_grid_mw=p_grid_mw,
        p_sol_mw=p_solar_mw,
        p_dis_mw=p_dis_mw,
        params=params,
        dt_hours=dt_hours,
    )

    # Calculate timing
    t_hours = optimizer_step * dt_hours
    timestamp_iso = None
    if step_datetime is not None:
        timestamp_iso = step_datetime.isoformat()
    elif hasattr(row, "timestamp") and not pd.isna(row["timestamp"]):
        timestamp_iso = pd.to_datetime(row["timestamp"]).isoformat()

    # Create detailed log entry
    log_entry = create_log_entry(
        step=optimizer_step,
        t_hours=t_hours,
        dt_hours=dt_hours,
        p_grid_mw=p_grid_mw,
        p_sol_mw=p_solar_mw,
        p_dis_mw=p_dis_mw,
        e_prev_mwh=e_prev_mwh,
        e_next_mwh=e_next_mwh,
        losses=losses,
        eta_ch=params.eta_ch,
        timestamp_iso=timestamp_iso,
    )

    # Add optimizer-specific fields
    log_entry.update(
        {
            "price_intraday": row["price_intraday"],
            "solar_total_MW": row["solar_MW"],
            "solar_direct_sell_MW": row["P_sol_sell_MW"],
            "profit_step": getattr(row, "profit_step", 0.0),
            "battery_mode": "Idle",  # Will be updated based on binary vars
        }
    )

    # Add binary decision variables if present
    for col in ["z_grid", "z_solbat", "z_dis", "q_flag", "s_dis", "cycle"]:
        if col in row:
            log_entry[col] = int(row[col])

    # Set battery mode based on binary variables
    if log_entry.get("z_grid", 0) > 0.5:
        log_entry["battery_mode"] = "Grid→Battery"
    elif log_entry.get("z_solbat", 0) > 0.5:
        log_entry["battery_mode"] = "Solar→Battery"
    elif log_entry.get("z_dis", 0) > 0.5:
        log_entry["battery_mode"] = "Battery→Grid"

    # Write to CSV - filter to only include standard battery module fields
    # Define standard battery module fields that write_simulation_csv accepts
    standard_battery_fields = {
        "simulation_step",
        "elapsed_time_hours",
        "time_step_hours",
        "grid_power_mw",
        "solar_power_mw",
        "discharge_power_mw",
        "energy_before_mwh",
        "energy_after_mwh",
        "energy_in_from_grid_mwh",
        "energy_in_from_solar_mwh",
        "energy_in_total_mwh",
        "energy_out_total_mwh",
        "loss_charging_inefficiency_mwh",
        "loss_discharge_inefficiency_mwh",
        "loss_auxiliary_power_mwh",
        "loss_self_discharge_mwh",
        "loss_total_mwh",
        "timestamp_iso",  # Optional field
    }

    # Filter log entry to only include standard battery fields
    filtered_log_entry = {k: v for k, v in log_entry.items() if k in standard_battery_fields}

    if append_mode and Path(csv_path).exists():
        # For append mode, we need to use the same CSV format as write_simulation_csv
        # Read existing headers to match format
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_headers = next(reader, [])

        # Append single row to existing CSV
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=existing_headers)
            writer.writerow(filtered_log_entry)
    else:
        # Create new CSV with headers using standard battery module function
        write_simulation_csv([filtered_log_entry], csv_path)

    # Return the original log entry with all fields for testing/debugging
    return log_entry
