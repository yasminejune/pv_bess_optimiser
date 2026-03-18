# PV + Price Input Builder --- Optimizer Feed Function

## Overview

`create_input_df()` is the primary entrypoint for generating the
combined 15-minute dataset used by the optimizer.

It connects:

-   PV generation forecast (`generate_pv_power_for_date_range`)
-   Price prediction model (`run_inference`)
-   Timestamp alignment and merging logic

into a single call that produces a unified DataFrame indexed by UTC
timestamps.

This function is intended to prepare the live input feed for downstream
optimization workflows.

------------------------------------------------------------------------

## Data Flow

PVSiteConfig\
↓\
generate_pv_power_for_date_range()\
↓\
PV DataFrame (timestamp_utc, generation_kw)\
↓\
run_inference(model_path=LGBM_MODEL_DIR, \*\*kwargs)\
↓\
Price DataFrame (Timestamp → timestamp_utc, Price_pred → price_intraday)\
↓\
Outer merge on timestamp_utc\
↓\
Column rename: timestamp_utc → timestamp, Price_pred → price_intraday\
↓\
Combined Optimizer Input DataFrame (96 rows, 15-min UTC)

------------------------------------------------------------------------

## What the Function Does

1.  Resolves the forecasting time window:

    -   Defaults to current UTC time rounded down to the previous
        15-minute boundary.
    -   Defaults to `start_datetime + 1 day` if no end time is provided.

2.  Computes the horizon length in hours.

3.  Generates 15-minute PV production forecast.

4.  Runs the price prediction model — defaults to the LGBM
    `ForecasterRecursive` in `LGBM_MODEL_DIR`; pass `model_path` to override.

5.  Merges PV and price data on `timestamp_utc` using an outer join.

6.  Renames columns: `timestamp_utc` → `timestamp`, `Price_pred` →
    `price_intraday`.

7.  Returns the first 96 rows (24 hours at 15-minute resolution).

------------------------------------------------------------------------

## Usage

### With a Predefined Site Configuration

``` python
from ors.services.optimizer.integration import create_input_df

config = get_pv_config()

df = create_input_df()
print(df.head())
```

------------------------------------------------------------------------

### With Explicit Start/End Times

``` python
from datetime import datetime, timezone

start = datetime(2026, 3, 1, 10, 0, tzinfo=timezone.utc)
end   = datetime(2026, 3, 2, 10, 0, tzinfo=timezone.utc)

df = create_input_df(
    config=config,
    start_datetime=start,
    end_datetime=end,
)
```

------------------------------------------------------------------------

### Passing Price Model Inputs

`model_path` selects the price model.  It defaults to `LGBM_MODEL_DIR`
(`models/price_prediction/lgbm_recursive_single_model`), so no argument is
needed for the standard LGBM workflow.

``` python
from pathlib import Path

# Default — uses the most recent LGBM .joblib in LGBM_MODEL_DIR
df = create_input_df(config=config)

# Explicit LGBM directory
df = create_input_df(
    config=config,
    model_path=Path("models/price_prediction/lgbm_recursive_single_model"),
)

# Specific LGBM .joblib file
df = create_input_df(
    config=config,
    model_path=Path("models/price_prediction/lgbm_recursive_single_model/recursive_single_model_20260306_132459.joblib"),
)

# Legacy XGBoost model
df = create_input_df(
    config=config,
    model_path=Path("models/price_prediction/model.pkl"),
    lag_steps=(1, 2, 3, 6, 12, 24),
)
```

Additional `**kwargs` are forwarded directly to `run_inference()`.

------------------------------------------------------------------------

## Function Signature

``` python
def create_input_df(
    config: PVSiteConfig = None,
    *,
    client: Any | None = None,
    start_datetime: datetime | None = None,
    end_datetime: datetime | None = None,
    model_path: Path = LGBM_MODEL_DIR,
    **kwargs,
) -> pd.DataFrame
```

------------------------------------------------------------------------

## Default Time Behavior

If no time arguments are provided:

-   `start_datetime` → current UTC time floored to previous 15-minute
    boundary\
-   `end_datetime` → `start_datetime + 24 hours`

All timestamps are forced to UTC.\
Naive datetimes are assumed to be UTC.

------------------------------------------------------------------------

## Merge Strategy

The function uses:

`how="outer"`

Always keeping until 96 entries (24 hours)

This ensures:

-   All PV timestamps are preserved
-   All price timestamps are preserved
-   Missing values appear as NaN where one dataset does not contain the
    timestamp


------------------------------------------------------------------------

## Output

A DataFrame of exactly 96 rows (24 hours at 15-minute resolution), sorted by
`timestamp`.

Typical structure:

  timestamp                   generation_kw   price_intraday
  --------------------------- --------------- ---------------
  2026-03-01 10:00:00+00:00   12500.0         85.12
  2026-03-01 10:15:00+00:00   14200.0         83.95

Key columns:

- `timestamp` — UTC-aware `datetime64[ns, UTC]`
- `generation_kw` — PV generation forecast in kW
- `price_intraday` — LGBM price forecast in £/MWh

------------------------------------------------------------------------

## Units Summary

  Layer                    Power   Time         Price
  ------------------------ ------- ------------ -----------------
  PV forecast output       kW      15-min UTC   ---
  Price inference output   ---     15-min UTC   model-dependent
  Final DataFrame          kW      15-min UTC   model-dependent

------------------------------------------------------------------------

## Limitations

-   If `run_inference()` generates a different horizon than the PV
    forecast, outer merge may produce partial rows with NaNs.
-   Timestamp alignment assumes both services operate at 15-minute
    resolution.
-   No automatic gap filling or interpolation is performed.
-   The LGBM model must be trained before calling `create_input_df`.
    Run `python -m ors.services.prediction.train_script --model-name lgbm_recursive`
    from the repo root if `models/price_prediction/lgbm_recursive_single_model/`
    is empty or missing.

------------------------------------------------------------------------

## Intended Use

This function is designed to:

-   Generate live optimizer inputs
-   Create aligned PV + price forecasts
-   Provide a unified dataset for arbitrage or scheduling optimization

It is not intended for historical backtesting unless explicit date
ranges are provided.
