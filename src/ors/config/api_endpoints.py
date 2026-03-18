"""Global API endpoint URLs loaded from endpoints.toml."""

from __future__ import annotations

import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[import-untyped]
    except ModuleNotFoundError:
        import tomli as tomllib  # type: ignore[import-not-found]

_CONFIG_PATH = Path(__file__).resolve().parent / "endpoints.toml"

with _CONFIG_PATH.open("rb") as _f:
    _config = tomllib.load(_f)

FORECAST_API_URL: str = _config["open_meteo"]["forecast_url"]
ARCHIVE_API_URL: str = _config["open_meteo"]["archive_url"]
