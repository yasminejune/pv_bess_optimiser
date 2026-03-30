"""Live price inference service: fetch live data, run ETL, predict prices."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from ors.services.price_inference.live_data_pipeline import (
    build_live_lgbm_dataset,
    build_live_merged_dataset,
)
from ors.services.price_inference.live_inference import (
    LGBM_MODEL_DIR,
    find_latest_lgbm_model,
    load_model,
    prepare_features_for_inference,
    run_inference,
    select_forecast_rows,
)


def _to_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC, assuming naive inputs are already UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def get_price_forecast_df(
    start_datetime: datetime,
    end_datetime: datetime,
    time_step_minutes: int = 15,
    *,
    model_path: Path = LGBM_MODEL_DIR,
    **kwargs: Any,
) -> pd.DataFrame:
    """Return a price forecast aligned to the requested optimization horizon."""
    start_utc = _to_utc(start_datetime)
    end_utc = _to_utc(end_datetime)

    if end_utc <= start_utc:
        raise ValueError("end_datetime must be after start_datetime")

    if "horizon_hours" not in kwargs:
        kwargs["horizon_hours"] = math.ceil((end_utc - start_utc).total_seconds() / 3600.0)

    price_df = run_inference(
        model_path=model_path,
        reference_time=start_utc,
        **kwargs,
    )

    if "Timestamp" not in price_df.columns or "Price_pred" not in price_df.columns:
        raise KeyError("Price forecast must contain 'Timestamp' and 'Price_pred' columns")

    expected_15 = pd.date_range(
        start=start_utc,
        end=end_utc,
        freq="15min",
        inclusive="left",
        tz="UTC",
    )

    price_series = pd.Series(
        pd.to_numeric(price_df["Price_pred"], errors="coerce").to_numpy(),
        index=pd.to_datetime(price_df["Timestamp"], utc=True),
        dtype="float64",
    ).sort_index()

    aligned_15 = (
        price_series.reindex(price_series.index.union(expected_15))
        .sort_index()
        .interpolate(method="time")
        .ffill()
        .bfill()
        .reindex(expected_15)
    )

    target_index = pd.date_range(
        start=start_utc,
        end=end_utc,
        freq=f"{time_step_minutes}min",
        inclusive="left",
        tz="UTC",
    )

    if time_step_minutes == 15:
        aligned = aligned_15
    else:
        aligned = aligned_15.resample(
            f"{time_step_minutes}min",
            origin=start_utc,
        ).mean()

    aligned = aligned.reindex(target_index)
    if aligned.isna().any():
        raise ValueError("Price forecast did not cover the requested optimization horizon")

    return pd.DataFrame(
        {
            "timestamp": target_index,
            "price_intraday": aligned.astype(float).to_numpy(),
        }
    )


def get_price_forecast(
    start_datetime: datetime,
    end_datetime: datetime,
    time_step_minutes: int = 15,
    *,
    model_path: Path = LGBM_MODEL_DIR,
    **kwargs: Any,
) -> list[float]:
    """Return a price forecast list aligned to the requested optimization horizon."""
    forecast_df = get_price_forecast_df(
        start_datetime,
        end_datetime,
        time_step_minutes,
        model_path=model_path,
        **kwargs,
    )
    return list(forecast_df["price_intraday"])


__all__ = [
    "build_live_lgbm_dataset",
    "build_live_merged_dataset",
    "get_price_forecast",
    "get_price_forecast_df",
    "find_latest_lgbm_model",
    "load_model",
    "prepare_features_for_inference",
    "run_inference",
    "select_forecast_rows",
]
