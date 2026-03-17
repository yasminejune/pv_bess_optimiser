"""Data loading utilities for optimization runs.

Handles loading price, solar generation, and weather data from various sources:
- Automatic forecasting
- Historical data files
- Manual data profiles
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from ..config.optimization_config import (
    OptimizationConfig,
    PVConfiguration,
)


class DataLoadingError(Exception):
    """Raised when data loading fails."""

    pass


def load_price_data(
    config: OptimizationConfig,
    start_datetime: datetime,
    end_datetime: datetime,
    time_step_minutes: int = 15,
) -> tuple[dict[int, float], float]:
    """Load price data based on configuration.

    Args:
        config (OptimizationConfig): Optimization configuration
        start_datetime (datetime): Start of optimization period
        end_datetime (datetime): End of optimization period
        time_step_minutes (int): Time step resolution in minutes

    Returns:
        tuple[dict[int, float], float]: Tuple of (price_dict, terminal_price)
        - price_dict: 1-indexed dict of prices for each timestep
        - terminal_price: Price for valuing terminal battery energy

    Raises:
        DataLoadingError: If price data cannot be loaded
    """
    try:
        if config.optimization.price_source == "forecast":
            return _load_forecasted_prices(start_datetime, end_datetime, time_step_minutes, config)
        elif config.optimization.price_source == "historical":
            return _load_historical_prices(
                config.optimization.historical_price_path,
                start_datetime,
                end_datetime,
                time_step_minutes,
                config,
            )
        elif config.optimization.price_source == "manual":
            return _load_manual_prices(
                config.optimization.manual_price_profile,
                start_datetime,
                end_datetime,
                time_step_minutes,
                config,
            )
        else:
            raise DataLoadingError(f"Unknown price source: {config.optimization.price_source}")

    except Exception as e:
        raise DataLoadingError(f"Failed to load price data: {e}") from e


def load_solar_data(
    config: OptimizationConfig,
    start_datetime: datetime,
    end_datetime: datetime,
    time_step_minutes: int = 15,
) -> dict[int, float]:
    """Load solar generation data based on configuration.

    Args:
        config (OptimizationConfig): Optimization configuration
        start_datetime (datetime): Start of optimization period
        end_datetime (datetime): End of optimization period
        time_step_minutes (int): Time step resolution in minutes

    Returns:
        dict[int, float]: 1-indexed dict of solar generation (MW) for each timestep

    Raises:
        DataLoadingError: If solar data cannot be loaded
    """
    if config.pv is None:
        # No PV system - return zero generation
        num_steps = int((end_datetime - start_datetime).total_seconds() / 60 / time_step_minutes)
        return dict.fromkeys(range(1, num_steps + 1), 0.0)

    try:
        if config.pv.generation_source == "forecast":
            return _load_forecasted_solar(
                config.pv, start_datetime, end_datetime, time_step_minutes
            )
        elif config.pv.generation_source == "historical":
            return _load_historical_solar(
                config.pv.historical_data_path, start_datetime, end_datetime, time_step_minutes
            )
        elif config.pv.generation_source == "manual":
            return _load_manual_solar(
                config.pv.manual_generation_profile, start_datetime, end_datetime, time_step_minutes
            )
        else:
            raise DataLoadingError(f"Unknown solar source: {config.pv.generation_source}")

    except Exception as e:
        raise DataLoadingError(f"Failed to load solar data: {e}") from e


# Price data loading functions


def _load_forecasted_prices(
    start_datetime: datetime,
    end_datetime: datetime,
    time_step_minutes: int,
    config: OptimizationConfig,
) -> tuple[dict[int, float], float]:
    """Load forecasted price data using ML inference service.

    Args:
        start_datetime (datetime): Start of forecast period
        end_datetime (datetime): End of forecast period
        time_step_minutes (int): Time step resolution in minutes
        config (OptimizationConfig): Optimization configuration

    Returns:
        tuple[dict[int, float], float]: Tuple of (price_dict, terminal_price)

    Raises:
        ImportError: If price inference service unavailable
    """
    try:
        # Try to use price inference service
        try:
            from ..services.price_inference import get_price_forecast  # type: ignore[attr-defined]
        except (ImportError, AttributeError):
            raise ImportError("Price inference service unavailable") from None

        price_data = get_price_forecast(start_datetime, end_datetime, time_step_minutes)

        # Convert to 1-indexed dict
        price_dict = {i + 1: float(price) for i, price in enumerate(price_data)}

        # Calculate terminal price
        terminal_price = _calculate_terminal_price(price_dict, config)

        return price_dict, terminal_price

    except ImportError:
        # Fallback to dummy data if price service unavailable
        print("Warning: Price inference service unavailable, using dummy price data")
        return _generate_dummy_prices(start_datetime, end_datetime, time_step_minutes, config)


def _load_historical_prices(
    csv_path: str | None,
    start_datetime: datetime,
    end_datetime: datetime,
    time_step_minutes: int,
    config: OptimizationConfig,
) -> tuple[dict[int, float], float]:
    """Load historical price data from CSV file.

    Args:
        csv_path (str | None): Path to CSV file containing price data
        start_datetime (datetime): Start of data period
        end_datetime (datetime): End of data period
        time_step_minutes (int): Time step resolution in minutes
        config (OptimizationConfig): Optimization configuration

    Returns:
        tuple[dict[int, float], float]: Tuple of (price_dict, terminal_price)

    Raises:
        DataLoadingError: If CSV file cannot be loaded or processed
    """
    if csv_path is None:
        raise DataLoadingError("Historical price path not provided")

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise DataLoadingError(f"Price CSV file not found: {csv_path}")

    # Load and filter CSV data
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Find required time period
    mask = (df["timestamp"] >= start_datetime) & (df["timestamp"] < end_datetime)
    period_data = df.loc[mask].copy()

    if len(period_data) == 0:
        raise DataLoadingError(f"No price data found for period {start_datetime} to {end_datetime}")

    # Convert to 1-indexed dict
    price_dict = {
        i + 1: float(row.price) for i, row in enumerate(period_data.itertuples(index=False))
    }

    # Calculate terminal price from historical data
    terminal_price = _calculate_terminal_price(price_dict, config, df)

    return price_dict, terminal_price


def _load_manual_prices(
    price_profile: dict[str, float] | None,
    start_datetime: datetime,
    end_datetime: datetime,
    time_step_minutes: int,
    config: OptimizationConfig,
) -> tuple[dict[int, float], float]:
    """Load manual price profile from user-defined configuration.

    Args:
        price_profile (dict[str, float] | None): Dictionary mapping time strings to prices
        start_datetime (datetime): Start of data period
        end_datetime (datetime): End of data period
        time_step_minutes (int): Time step resolution in minutes
        config (OptimizationConfig): Optimization configuration

    Returns:
        tuple[dict[int, float], float]: Tuple of (price_dict, terminal_price)

    Raises:
        DataLoadingError: If price profile is invalid or cannot be processed
    """
    if price_profile is None:
        raise DataLoadingError("Manual price profile not provided")

    # Convert profile to timestamp-indexed data
    profile_data = []
    for time_str, price in price_profile.items():
        hour, minute = map(int, time_str.split(":"))
        daily_time = start_datetime.replace(hour=hour, minute=minute)
        profile_data.append((daily_time, price))

    # Sort by time
    profile_data.sort(key=lambda x: x[0])

    # Interpolate to required resolution
    price_dict = _interpolate_hourly_to_timesteps(
        profile_data, start_datetime, end_datetime, time_step_minutes
    )

    # Use average price as terminal price
    terminal_price = sum(price_dict.values()) / len(price_dict)
    if config.optimization.terminal_price_gbp_per_mwh is not None:
        terminal_price = config.optimization.terminal_price_gbp_per_mwh

    return price_dict, terminal_price


# Solar data loading functions


def _load_forecasted_solar(
    pv_config: PVConfiguration,
    start_datetime: datetime,
    end_datetime: datetime,
    time_step_minutes: int,
) -> dict[int, float]:
    """Load forecasted solar generation data using weather services.

    Args:
        pv_config (PVConfiguration): PV system configuration
        start_datetime (datetime): Start of forecast period
        end_datetime (datetime): End of forecast period
        time_step_minutes (int): Time step resolution in minutes

    Returns:
        dict[int, float]: 1-indexed dict of solar generation (MW) for each timestep

    Raises:
        DataLoadingError: If location coordinates are missing
        ImportError: If weather/PV service unavailable
    """
    if pv_config.location_lat is None or pv_config.location_lon is None:
        raise DataLoadingError("Location coordinates required for solar forecasting")

    try:
        # Try to use weather/PV generation service
        try:
            from ..services.weather_to_pv import get_pv_forecast  # type: ignore[attr-defined]
        except (ImportError, AttributeError):
            raise ImportError("Weather/PV service unavailable") from None

        generation_data = get_pv_forecast(
            lat=pv_config.location_lat,
            lon=pv_config.location_lon,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            rated_power_kw=pv_config.rated_power_kw,
            panel_area_m2=pv_config.panel_area_m2 or 1000.0,
            panel_efficiency=pv_config.panel_efficiency,
        )

        # Convert kW to MW and create 1-indexed dict
        solar_dict = {i + 1: float(gen / 1000.0) for i, gen in enumerate(generation_data)}

        return solar_dict

    except ImportError:
        # Fallback to dummy solar pattern
        print("Warning: Solar forecast service unavailable, using dummy solar pattern")
        return _generate_dummy_solar(pv_config, start_datetime, end_datetime, time_step_minutes)


def _load_historical_solar(
    csv_path: str | None, start_datetime: datetime, end_datetime: datetime, time_step_minutes: int
) -> dict[int, float]:
    """Load historical solar generation data from CSV file.

    Args:
        csv_path (str | None): Path to CSV file containing solar generation data
        start_datetime (datetime): Start of data period
        end_datetime (datetime): End of data period
        time_step_minutes (int): Time step resolution in minutes

    Returns:
        dict[int, float]: 1-indexed dict of solar generation (MW) for each timestep

    Raises:
        DataLoadingError: If CSV file cannot be loaded or processed
    """
    if csv_path is None:
        raise DataLoadingError("Historical solar data path not provided")

    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise DataLoadingError(f"Solar CSV file not found: {csv_path}")

    # Load and filter CSV data
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Find required time period
    mask = (df["timestamp"] >= start_datetime) & (df["timestamp"] < end_datetime)
    period_data = df.loc[mask].copy()

    if len(period_data) == 0:
        raise DataLoadingError(f"No solar data found for period {start_datetime} to {end_datetime}")

    # Convert kW to MW and create 1-indexed dict
    generation_column = "generation_kw" if "generation_kw" in df.columns else "generation"
    solar_dict = {
        i + 1: float(row[generation_column] / 1000.0)
        for i, (_, row) in enumerate(period_data.iterrows())
    }

    return solar_dict


def _load_manual_solar(
    generation_profile: dict[str, float] | None,
    start_datetime: datetime,
    end_datetime: datetime,
    time_step_minutes: int,
) -> dict[int, float]:
    """Load manual solar generation profile from user configuration.

    Args:
        generation_profile (dict[str, float] | None): Dictionary mapping time strings to generation (kW)
        start_datetime (datetime): Start of data period
        end_datetime (datetime): End of data period
        time_step_minutes (int): Time step resolution in minutes

    Returns:
        dict[int, float]: 1-indexed dict of solar generation (MW) for each timestep

    Raises:
        DataLoadingError: If generation profile is invalid or cannot be processed
    """
    if generation_profile is None:
        raise DataLoadingError("Manual generation profile not provided")

    # Convert profile to timestamp-indexed data
    profile_data = []
    for time_str, generation_kw in generation_profile.items():
        hour, minute = map(int, time_str.split(":"))
        daily_time = start_datetime.replace(hour=hour, minute=minute)
        # Convert kW to MW
        generation_mw = generation_kw / 1000.0
        profile_data.append((daily_time, generation_mw))

    # Sort by time
    profile_data.sort(key=lambda x: x[0])

    # Interpolate to required resolution
    solar_dict = _interpolate_hourly_to_timesteps(
        profile_data, start_datetime, end_datetime, time_step_minutes
    )

    return solar_dict


# Utility functions


def _calculate_terminal_price(
    price_dict: dict[int, float],
    config: OptimizationConfig,
    historical_df: pd.DataFrame | None = None,
) -> float:
    """Calculate terminal price for battery energy valuation.

    Args:
        price_dict (dict[int, float]): 1-indexed price data for optimization period
        config (OptimizationConfig): Optimization configuration
        historical_df (pd.DataFrame | None): Optional historical data for averaging

    Returns:
        float: Terminal price in GBP/MWh

    Raises:
        DataLoadingError: If terminal price method is invalid or required data is missing
    """
    if config.optimization.terminal_price_method == "manual":
        if config.optimization.terminal_price_gbp_per_mwh is None:
            raise DataLoadingError("Manual terminal price required but not provided")
        return config.optimization.terminal_price_gbp_per_mwh

    elif config.optimization.terminal_price_method == "average":
        if historical_df is not None:
            # Use 30-day historical average if available
            return float(historical_df["price"].tail(30 * 96).mean())  # 30 days * 96 steps/day
        else:
            # Use average from optimization period
            return sum(price_dict.values()) / len(price_dict)

    else:
        raise DataLoadingError(
            f"Unknown terminal price method: {config.optimization.terminal_price_method}"
        )


def _interpolate_hourly_to_timesteps(
    hourly_data: list[tuple[datetime, float]],
    start_datetime: datetime,
    end_datetime: datetime,
    time_step_minutes: int,
) -> dict[int, float]:
    """Interpolate hourly data to required time step resolution.

    Args:
        hourly_data (list[tuple[datetime, float]]): Hourly timestamp-value pairs
        start_datetime (datetime): Start of interpolation period
        end_datetime (datetime): End of interpolation period
        time_step_minutes (int): Target time step resolution in minutes

    Returns:
        dict[int, float]: 1-indexed dict of interpolated values
    """
    # Create timestamp series for required resolution
    timestamps = pd.date_range(
        start=start_datetime, end=end_datetime, freq=f"{time_step_minutes}min", inclusive="left"
    )

    # Interpolate to required resolution
    interpolated_series = pd.Series(
        index=pd.to_datetime([t for t, v in hourly_data]), data=[v for t, v in hourly_data]
    )

    # Resample to target frequency
    resampled = interpolated_series.resample(f"{time_step_minutes}min").interpolate(method="linear")

    # Map to timestamps and convert to 1-indexed dict
    result_dict = {}
    for i, ts in enumerate(timestamps):
        if ts in resampled.index:
            result_dict[i + 1] = float(resampled[ts])
        else:
            # Use nearest value if exact timestamp not found
            nearest_val = interpolated_series.iloc[
                interpolated_series.index.get_indexer([ts], method="nearest")[0]
            ]
            result_dict[i + 1] = float(nearest_val)

    return result_dict


def _generate_dummy_prices(
    start_datetime: datetime,
    end_datetime: datetime,
    time_step_minutes: int,
    config: OptimizationConfig,
) -> tuple[dict[int, float], float]:
    """Generate dummy price data for testing purposes.

    Args:
        start_datetime (datetime): Start of data period
        end_datetime (datetime): End of data period
        time_step_minutes (int): Time step resolution in minutes
        config (OptimizationConfig): Optimization configuration

    Returns:
        tuple[dict[int, float], float]: Tuple of (price_dict, terminal_price)
    """
    import math

    num_steps = int((end_datetime - start_datetime).total_seconds() / 60 / time_step_minutes)
    price_dict = {}

    # Generate realistic daily price pattern
    for i in range(1, num_steps + 1):
        # Hour of day (0-24)
        hour = ((i - 1) * time_step_minutes / 60) % 24

        # Base price with daily pattern
        base_price = 50.0
        daily_variation = 20.0 * math.sin(2 * math.pi * (hour - 6) / 24)  # Peak around 6 PM
        morning_peak = 10.0 * math.exp(-((hour - 8) ** 2) / 8)  # Morning peak
        evening_peak = 15.0 * math.exp(-((hour - 19) ** 2) / 8)  # Evening peak

        price = base_price + daily_variation + morning_peak + evening_peak
        price_dict[i] = max(price, 20.0)  # Minimum price 20 £/MWh

    # Average price as terminal price
    terminal_price = sum(price_dict.values()) / len(price_dict)

    return price_dict, terminal_price


def _generate_dummy_solar(
    pv_config: PVConfiguration,
    start_datetime: datetime,
    end_datetime: datetime,
    time_step_minutes: int,
) -> dict[int, float]:
    """Generate dummy solar generation data for testing purposes.

    Args:
        pv_config (PVConfiguration): PV system configuration
        start_datetime (datetime): Start of data period
        end_datetime (datetime): End of data period
        time_step_minutes (int): Time step resolution in minutes

    Returns:
        dict[int, float]: 1-indexed dict of solar generation (MW) for each timestep
    """
    import math

    num_steps = int((end_datetime - start_datetime).total_seconds() / 60 / time_step_minutes)
    solar_dict = {}

    # Convert rated power from kW to MW
    rated_power_mw = pv_config.rated_power_kw / 1000.0

    for i in range(1, num_steps + 1):
        # Hour of day (0-24)
        hour = ((i - 1) * time_step_minutes / 60) % 24

        # Solar generation pattern (sunrise ~6 AM to sunset ~8 PM)
        if 6 <= hour <= 20:
            # Bell curve peaking at noon
            normalized_hour = (hour - 13) / 7  # Center at 1 PM, spread ±7 hours
            generation_factor = math.exp(-0.5 * (normalized_hour**2))
        else:
            generation_factor = 0.0

        solar_dict[i] = rated_power_mw * generation_factor

    return solar_dict
