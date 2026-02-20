
# BESS Intraday Optimizer
## Δt = 15 minutes · 6h Energy Capacity · Max 3 Cycles per Day

# Weather API Ingest (Open-Meteo UKMO)

## Purpose

This module fetches weather data from the Open-Meteo API (UK Met Office model) and parses it into clean, consistent schemas for downstream analytics and modelling.

It supports:

- Current conditions
- Forecast hourly data
- Forecast daily summaries
- Historical hourly data (Archive API)
- Historical daily summaries (Archive API)

All outputs follow a consistent, structured schema suitable for machine learning training or statistical analysis.


---

## Fixed Inputs (Deterministic Example)

The script uses fixed parameters defined in `DEFAULT_PARAMS`:

- Latitude: 54.7584
- Longitude: -2.6953
- Model: ukmo_uk_deterministic_2km
- Timezone: GMT
- Forecast hours: 24
- Past hours: 6
- Forecast days: 1

Historical example range:

- Start date: 2025-01-01
- End date: 2025-12-31

Hourly variables are defined in:

DEFAULT_PARAMS["hourly"]

Daily variables are defined in:

DEFAULT_PARAMS["daily"]

Current variables are defined in:

DEFAULT_PARAMS["current"]

---

## Module Functions

### build_client()

Creates an Open-Meteo client with:
- HTTP caching (1 hour)
- Retry logic (5 retries with backoff)

---

### fetch_api_response(client, params, url)

Fetches forecast data from the API.

Returns:
- First response object

Raises:
- ValueError if no responses are returned

---

### fetch_historical_hourly(...)

Fetches historical hourly data from the Open-Meteo Archive API.

Returns:
- Pandas DataFrame using the same schema as forecast hourly data

---

### fetch_historical_daily(...)

Fetches historical daily data from the Archive API.

Returns:
- Pandas DataFrame using the same schema as forecast daily data

---

### parse_current(api_response, current_vars) → dict

Returns a dictionary containing:
- time_unix
- time_utc
- All variables listed in current_vars

---

### parse_hourly(api_response, hourly_vars) → DataFrame

Returns a DataFrame containing:
- timestamp_utc
- All variables listed in hourly_vars

Raises:
- WeatherIngestError if any variable length does not match timestamp length

Allows:
- Missing values (NaN), which may occur in historical datasets

---

### parse_daily(api_response, daily_vars) → DataFrame

Returns a DataFrame containing:
- date_utc
- All variables listed in daily_vars

Special handling:
- sunrise and sunset are converted to UTC datetime

---

## Output Schemas

### Hourly DataFrame

Columns:
- timestamp_utc
- plus all variables in DEFAULT_PARAMS["hourly"]

Example output file:

data/historical_hourly_2025.csv

Expected rows for 1 year: approximately 8760

---

### Daily DataFrame

Columns:
- date_utc
- plus all variables in DEFAULT_PARAMS["daily"]

Example output file:

data/historical_daily_2025.csv

Expected rows for 1 year: 365

---

### Current Conditions (dict)

Keys:
- time_unix
- time_utc
- plus all variables in DEFAULT_PARAMS["current"]

---

## Historical Data Notes

Historical data is fetched from:

https://archive-api.open-meteo.com/v1/archive

Schema is identical to forecast schema.

Some variables may contain missing values (NaN) depending on:
- Location
- Model availability
- Time period coverage

The parser intentionally allows NaN values to ensure robust downstream processing.

---

## Reference Request (Validation URL)

The following URL was used to validate selected variables and parameters:

https://archive-api.open-meteo.com/v1/archive?latitude=54.7584&longitude=-2.6953&start_date=2025-01-01&end_date=2025-12-31&daily=daylight_duration,shortwave_radiation_sum,sunrise,sunset,precipitation_sum,rain_sum,snowfall_sum,precipitation_hours,weather_code,temperature_2m_max,temperature_2m_min&hourly=temperature_2m,is_day,precipitation,rain,cloud_cover_low,cloud_cover_mid,cloud_cover_high,relative_humidity_2m,dew_point_2m,wind_speed_10m,wind_gusts_10m,wind_direction_10m,snowfall,surface_pressure,weather_code,sunshine_duration,shortwave_radiation,direct_radiation,cloud_cover,diffuse_radiation&timezone=GMT

---

## How to Run

From the project directory:

python3 weather_fetcher.py

This will:

1. Fetch current and forecast weather data
2. Fetch full 2025 historical hourly and daily data
3. Create the following files inside the `data/` folder:
   - data/historical_hourly_2025.csv
   - data/historical_daily_2025.csv
4. Print row counts in the terminal for verification
---

## How to Run Tests

python3 -m pytest -q

---

## 9) Explicit assumptions
- Solar power S_t ≤ 100 MW  
- Solar cannot be exported directly  
- No ramp-rate limits  
- No degradation cost (only cycle count)  
- No thermal derating or forced outages  

---

## Platform prerequisites and installation notes

This project is intended to run on Windows, macOS, and Linux. A few external/tooling prerequisites are platform-specific — follow the steps below for a smooth install.


If you prefer pip, prefer pip installing wheels in a virtualenv:

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\Activate     # Windows (PowerShell: .\.venv\Scripts\Activate.ps1)

pip install --upgrade pip
pip install -e .[dev,ml]    # avoid `solvers` unless you installed system solvers (see below)
```

Solver notes (important) - Read the following only if the above is insufficient in satisfying all dependencies. 
- GLPK (default solver in `pyproject.toml`): the pyomo `glpk` backend expects the GLPK executable (`glpsol`) on PATH.
  - macOS (Homebrew): `brew install glpk`
  - Debian/Ubuntu: `sudo apt-get update && sudo apt-get install -y glpk-utils libglpk-dev`
  - Windows: install via conda (`conda install -c conda-forge glpk`) or download a prebuilt binary and add it to PATH. Using WSL is an alternate route.
- HiGHS (`highspy`): available as wheels on many platforms; prefer conda-forge if pip wheel is unavailable.
- NEOS solver manager: the project defaults to `neos` in `pyproject.toml`. NEOS requires network access; if you are offline or behind a firewall, change `solver_manager` to `local` and configure a local solver.

Troubleshooting tips
- If pip build fails for `numpy`/`pandas`/`highspy`, install prebuilt packages via conda or use a Python version with available wheels (e.g., 3.10–3.12).
- To avoid the `glpk` executable requirement, change the default solver in your runtime (or in `pyproject.toml`) to a solver you have installed locally.

Packaging note
- The `all` extra previously referenced an undefined `ors` extra; this has been removed. Use `pip install -e .[all]` after installing any required system solvers, or install extras selectively (e.g., `pip install -e .[dev,ml]`).

All core parsing logic is covered by automated tests.


---

## Code Quality

The project enforces **formatting** (Black), **linting + docstring style** (Ruff), and **type checking** (mypy) across all source code.

### Running locally

```bash
# Auto-format with Black
make format

# Lint with Ruff (auto-fix safe issues)
make lint

# Type-check with mypy
make typecheck

# Run all checks together (same as CI – exits non-zero on failure)
make ci
```

Or without `make`:

```bash
python -m black src/ tests/
python -m ruff check src/ tests/ --fix
python -m mypy src/
```

### Rules enforced

| Tool  | What is checked |
|-------|-----------------|
| **Black** | Consistent formatting, line length 100 |
| **Ruff D** (pydocstyle, Google convention) | Public functions and classes must have Google-style docstrings with `Args:`, `Returns:`, `Raises:`, and `Attributes:` sections where applicable |
| **Ruff ANN** (flake8-annotations) | All function arguments and return types must be explicitly annotated |
| **mypy** | Full type-checking with `disallow_untyped_defs`, `disallow_incomplete_defs`, `no_implicit_optional`, and `warn_return_any` |

### Conventions

- **Docstring style**: Google style – summary on the opening line, followed by `Args:`, `Returns:`, and `Raises:` sections as needed.
- **Private helpers** (leading `_`): type annotations are still required; docstrings are optional.
- **Tests** and the Pyomo **optimizer script** are excluded from docstring/annotation checks (see `pyproject.toml` `per-file-ignores`).

### CI enforcement

The `make ci` command is the authoritative check. All three tools must exit 0:

```
black --check src/ tests/    # formatting
ruff check src/ tests/       # lint + docstrings + annotations
mypy src/                    # type checking
```
