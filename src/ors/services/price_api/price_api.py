# price_api.py


from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd
import requests
from dateutil import parser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# -------------------------
# Session / HTTP
# -------------------------
def make_session() -> requests.Session:
    """Create a reusable requests.Session with retry strategy.

    Returns:
        requests.Session: Session configured with retries for transient errors.
    """
    s = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=20, pool_maxsize=20)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _normalize_payload(payload: Any) -> pd.DataFrame:
    """Normalize a BMRS JSON payload to a pandas DataFrame.

    BMRS responses can come as:
      - list
      - dict with keys like data/results/items/records
      - generic dict fallback

    Args:
        payload: Parsed JSON payload (list/dict/etc).

    Returns:
        pd.DataFrame: Normalized DataFrame (possibly empty).
    """
    if isinstance(payload, list):
        return pd.json_normalize(payload)

    if isinstance(payload, dict):
        for k in ("data", "results", "result", "items", "records"):
            v = payload.get(k)
            if isinstance(v, list):
                return pd.json_normalize(v)

        # fallback: first list-valued entry
        for v in payload.values():
            if isinstance(v, list):
                return pd.json_normalize(v)

        return pd.json_normalize([payload])

    return pd.DataFrame()


def bmrs_get(
    session: requests.Session,
    base_url: str,
    path: str,
    params: dict,
    timeout: int = 60,
) -> pd.DataFrame:
    """Perform a GET request against the BMRS API and normalize JSON response.

    Args:
        session: Requests session.
        base_url: Base URL (e.g., "https://data.elexon.co.uk/bmrs/api/v1").
        path: Endpoint path (e.g., "/datasets/MID").
        params: Query parameters.
        timeout: Request timeout (seconds).

    Returns:
        pd.DataFrame: Normalized response.

    Raises:
        requests.HTTPError: If the API returns >= 400.
    """
    url = f"{base_url}{path}"
    r = session.get(url, params=params, timeout=timeout)
    if r.status_code >= 400:
        raise requests.HTTPError(f"{r.status_code} | {url} | {r.text[:300]}")
    return _normalize_payload(r.json())


# -------------------------
# Time helpers
# -------------------------
def to_dt_utc(x: str | datetime) -> datetime:
    """Convert an ISO string or datetime to an aware UTC datetime.

    Args:
        x: ISO-8601 string or datetime.

    Returns:
        datetime: UTC-aware datetime.
    """
    if isinstance(x, datetime):
        return x.astimezone(timezone.utc)
    return parser.isoparse(x).astimezone(timezone.utc)


def to_isoz(dt: datetime) -> str:
    """Format datetime as full ISO string ending with Z.

    Args:
        dt: datetime value.

    Returns:
        str: ISO-8601 in UTC with trailing 'Z'.
    """
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def to_rfc3339_minute_z(dt: datetime) -> str:
    """Format datetime as RFC3339 minute precision: YYYY-MM-DDTHH:MMZ.

    Args:
        dt: datetime value.

    Returns:
        str: RFC3339 minute-precision timestamp in UTC.
    """
    dt_utc = dt.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return dt_utc.strftime("%Y-%m-%dT%H:%MZ")


# -------------------------
# Param styles + robust getter
# -------------------------
def _params_from_to(start_utc: datetime, end_utc: datetime) -> dict:
    return {
        "format": "json",
        "from": to_rfc3339_minute_z(start_utc),
        "to": to_rfc3339_minute_z(end_utc),
    }


def _params_publish_window(start_utc: datetime, end_utc: datetime) -> dict:
    return {
        "format": "json",
        "publishDateTimeFrom": to_isoz(start_utc),
        "publishDateTimeTo": to_isoz(end_utc),
    }


def _params_settlement_window(start_utc: datetime, end_utc: datetime) -> dict:
    return {
        "format": "json",
        "settlementDateFrom": start_utc.strftime("%Y-%m-%d"),
        "settlementDateTo": end_utc.strftime("%Y-%m-%d"),
    }


def bmrs_get_first_success(
    session: requests.Session,
    base_url: str,
    path: str,
    params_candidates: list[dict],
    timeout: int = 60,
) -> pd.DataFrame:
    """Try multiple parameter styles and return the first non-empty successful response.

    Some endpoints return HTTP 200 with empty data for unsupported parameter styles.
    This helper continues trying candidates until it finds a non-empty response.

    Args:
        session: Requests session.
        base_url: Base URL.
        path: Endpoint path.
        params_candidates: List of query-param dictionaries to try.
        timeout: Request timeout (seconds).

    Returns:
        pd.DataFrame: Normalized DataFrame (may be empty if all candidates are empty).

    Raises:
        Exception: Re-raises the last exception if all candidates error.
    """
    last_err: Exception | None = None
    first_empty: pd.DataFrame | None = None

    for params in params_candidates:
        try:
            df = bmrs_get(session, base_url, path, params, timeout=timeout)

            # Keep the first empty response as fallback, but continue trying
            if df.empty:
                if first_empty is None:
                    first_empty = df
                continue

            return df

        except Exception as ex:  # noqa: BLE001
            last_err = ex
            continue

    # If we never got a non-empty, return an empty (if we got one), else raise last error
    if first_empty is not None:
        return first_empty

    raise last_err if last_err else RuntimeError("All parameter candidates failed.")


# -------------------------
# Shared time extraction
# -------------------------
def _extract_ts_utc(raw: pd.DataFrame) -> pd.Series:
    """Extract a consistent UTC timestamp from BMRS records.

    Priority:
    1) settlementDate + settlementPeriod (most consistent across BMRS half-hourly datasets)
    2) settlementDateTime / startTime / publishDateTime / timestamp fallback
    """
    # 1) Best: settlement-based clock (half-hour periods)
    if {"settlementDate", "settlementPeriod"}.issubset(raw.columns):
        d = pd.to_datetime(raw["settlementDate"], errors="coerce")
        sp = pd.to_numeric(raw["settlementPeriod"], errors="coerce")

        local = d.dt.tz_localize(
            "Europe/London", ambiguous="NaT", nonexistent="shift_forward"
        ) + pd.to_timedelta((sp - 1) * 30, unit="m")

        return local.dt.tz_convert("UTC")

    # 2) Fallbacks if settlement fields aren't present
    for c in (
        "settlementDateTime",
        "startTime",
        "publishDateTime",
        "publishTime",
        "timestamp",
        "dateTime",
        "time",
    ):
        if c in raw.columns:
            ts = pd.to_datetime(raw[c], utc=True, errors="coerce")
            if ts.notna().any():
                return ts

    return pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")


def _extract_ts_from_settlement(raw: pd.DataFrame) -> pd.Series:
    """Build ts_utc from settlementDate + settlementPeriod (half-hour GB local time)."""
    if not {"settlementDate", "settlementPeriod"}.issubset(raw.columns):
        return pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")

    d = pd.to_datetime(raw["settlementDate"], errors="coerce")
    sp = pd.to_numeric(raw["settlementPeriod"], errors="coerce")

    local = d.dt.tz_localize(
        "Europe/London", ambiguous="NaT", nonexistent="shift_forward"
    ) + pd.to_timedelta((sp - 1) * 30, unit="m")
    return local.dt.tz_convert("UTC")


def _first_existing_column(raw: pd.DataFrame, candidates: list[str]) -> pd.Series:
    """Return the first existing column from candidates, or an all-NA series."""
    for c in candidates:
        if c in raw.columns:
            return raw[c]
    return pd.Series([pd.NA] * len(raw), index=raw.index)


# -------------------------
# Endpoint-specific calls
# (each returns a STRUCTURED frame with ts_utc + value column)
# -------------------------
def fetch_mid_price(
    session: requests.Session,
    base_url: str,
    start_utc: datetime,
    end_utc: datetime,
    data_providers: list[str],
    timeout: int = 60,
    path: str = "/datasets/MID",
) -> pd.DataFrame:
    """Fetch MID price.

    Args:
        session: Requests session.
        base_url: Base URL.
        start_utc: Start datetime (UTC).
        end_utc: End datetime (UTC).
        data_providers: Provider filter list (may be empty for no filter).
        timeout: Request timeout.
        path: Endpoint path.

    Returns:
        pd.DataFrame: Columns [ts_utc, price].
    """
    params = _params_from_to(start_utc, end_utc)
    if data_providers:
        params["dataProviders"] = data_providers

    raw = bmrs_get(session, base_url, path, params, timeout=timeout)
    if raw.empty:
        return pd.DataFrame(columns=["ts_utc", "price"])

    ts = _extract_ts_utc(raw)
    if ts.isna().all():
        return pd.DataFrame(columns=["ts_utc", "price"])

    out = pd.DataFrame({"ts_utc": ts})
    out["price"] = pd.to_numeric(raw.get("price"), errors="coerce")
    out = out.dropna(subset=["ts_utc"]).sort_values("ts_utc")
    out = out.drop_duplicates(subset=["ts_utc"], keep="last")
    return out


def fetch_itsdo_demand(
    session: requests.Session,
    base_url: str,
    start_utc: datetime,
    end_utc: datetime,
    timeout: int = 60,
    path: str = "/datasets/ITSDO",
) -> pd.DataFrame:
    """Fetch ITSDO demand.

    Args:
        session: Requests session.
        base_url: Base URL.
        start_utc: Start datetime (UTC).
        end_utc: End datetime (UTC).
        timeout: Request timeout.
        path: Endpoint path.

    Returns:
        pd.DataFrame: Columns [ts_utc, demand_itsdo].
    """
    params_candidates = [
        _params_publish_window(start_utc, end_utc),
        _params_settlement_window(start_utc, end_utc),
        _params_from_to(start_utc, end_utc),
    ]

    raw = bmrs_get_first_success(
        session=session,
        base_url=base_url,
        path=path,
        params_candidates=params_candidates,
        timeout=timeout,
    )

    if raw.empty:
        return pd.DataFrame(columns=["ts_utc", "demand_itsdo"])

    ts = _extract_ts_from_settlement(raw)
    if ts.isna().all():
        ts = _extract_ts_utc(raw)

    if ts.isna().all():
        return pd.DataFrame(columns=["ts_utc", "demand_itsdo"])
    ts = _extract_ts_from_settlement(raw)

    out = pd.DataFrame({"ts_utc": ts})
    out["demand_itsdo"] = pd.to_numeric(
        _first_existing_column(
            raw, ["demand", "transmissionSystemDemandOutturn", "itsdo", "value"]
        ),
        errors="coerce",
    )
    out = out.dropna(subset=["ts_utc"]).sort_values("ts_utc")
    out = out.drop_duplicates(subset=["ts_utc"], keep="last")
    return out


def fetch_indo_initial_demand(
    session: requests.Session,
    base_url: str,
    start_utc: datetime,
    end_utc: datetime,
    timeout: int = 60,
    path: str = "/datasets/INDO",
) -> pd.DataFrame:
    """Fetch INDO initial demand outturn.

    Args:
        session: Requests session.
        base_url: Base URL.
        start_utc: Start datetime (UTC).
        end_utc: End datetime (UTC).
        timeout: Request timeout.
        path: Endpoint path.

    Returns:
        pd.DataFrame: Columns [ts_utc, demand_indo].
    """
    params_candidates = [
        _params_publish_window(start_utc, end_utc),
        _params_settlement_window(start_utc, end_utc),
        _params_from_to(start_utc, end_utc),
    ]

    raw = bmrs_get_first_success(
        session=session,
        base_url=base_url,
        path=path,
        params_candidates=params_candidates,
        timeout=timeout,
    )
    if raw.empty:
        return pd.DataFrame(columns=["ts_utc", "demand_indo"])

    ts = _extract_ts_from_settlement(raw)
    if ts.isna().all():
        ts = _extract_ts_utc(raw)

    if ts.isna().all():
        return pd.DataFrame(columns=["ts_utc", "demand_indo"])

    out = pd.DataFrame({"ts_utc": ts})
    out["demand_indo"] = pd.to_numeric(
        _first_existing_column(raw, ["initialDemandOutturn", "demand", "indo", "value"]),
        errors="coerce",
    )
    out = out.dropna(subset=["ts_utc"]).sort_values("ts_utc")
    out = out.drop_duplicates(subset=["ts_utc"], keep="last")
    return out


def fetch_inddem_demand(
    session: requests.Session,
    base_url: str,
    start_utc: datetime,
    end_utc: datetime,
    timeout: int = 60,
    path: str = "/datasets/INDDEM",
) -> pd.DataFrame:
    """Fetch INDDEM demand.

    Args:
        session: Requests session.
        base_url: Base URL.
        start_utc: Start datetime (UTC).
        end_utc: End datetime (UTC).
        timeout: Request timeout.
        path: Endpoint path.

    Returns:
        pd.DataFrame: Columns [ts_utc, demand_inddem].
    """
    params_candidates = [
        _params_publish_window(start_utc, end_utc),
        _params_settlement_window(start_utc, end_utc),
        _params_from_to(start_utc, end_utc),
    ]

    raw = bmrs_get_first_success(
        session=session,
        base_url=base_url,
        path=path,
        params_candidates=params_candidates,
        timeout=timeout,
    )
    if raw.empty:
        return pd.DataFrame(columns=["ts_utc", "demand_inddem"])

    ts = _extract_ts_from_settlement(raw)
    if ts.isna().all():
        ts = _extract_ts_utc(raw)

    if ts.isna().all():
        return pd.DataFrame(columns=["ts_utc", "demand_inddem"])

    out = pd.DataFrame({"ts_utc": ts})
    out["demand_inddem"] = pd.to_numeric(
        _first_existing_column(raw, ["demand", "indicatedDemand", "inddem", "value"]),
        errors="coerce",
    )

    out = out.dropna(subset=["ts_utc"]).sort_values("ts_utc")
    out = out.drop_duplicates(subset=["ts_utc"], keep="last")
    return out


# -------------------------
# Convenience: fetch "current" dataset (today 00:00 UTC -> now)
# -------------------------
def fetch_current_price_data(
    session: requests.Session,
    base_url: str,
    data_providers: list[str],
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch BMRS data for today (UTC) and return a price_data.csv-style DataFrame.

    Output columns:
        - price
        - demand_itsdo
        - timestamp
        - demand_indo
        - demand_inddem

    Args:
        session: Requests session.
        base_url: Base URL.
        data_providers: MID provider filters (may be empty list).
        timeout: Request timeout (seconds).

    Returns:
        pd.DataFrame: Final merged dataset.
    """
    now = datetime.now(timezone.utc)
    start_today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    mid = fetch_mid_price(
        session=session,
        base_url=base_url,
        start_utc=start_today,
        end_utc=now,
        data_providers=data_providers,
        timeout=timeout,
    )
    itsdo = fetch_itsdo_demand(
        session=session,
        base_url=base_url,
        start_utc=start_today,
        end_utc=now,
        timeout=timeout,
    )
    indo = fetch_indo_initial_demand(
        session=session,
        base_url=base_url,
        start_utc=start_today,
        end_utc=now,
        timeout=timeout,
    )
    inddem = fetch_inddem_demand(
        session=session,
        base_url=base_url,
        start_utc=start_today,
        end_utc=now,
        timeout=timeout,
    )

    def _merge(a: pd.DataFrame | None, b: pd.DataFrame) -> pd.DataFrame:
        if a is None or a.empty:
            return b
        if b.empty:
            return a
        return a.merge(b, on="ts_utc", how="outer")

    merged: pd.DataFrame | None = None
    merged = _merge(merged, mid)
    merged = _merge(merged, itsdo)
    merged = _merge(merged, indo)
    merged = _merge(merged, inddem)

    if merged is None or merged.empty:
        return pd.DataFrame(
            columns=[
                "price",
                "demand_itsdo",
                "timestamp",
                "demand_indo",
                "demand_inddem",
            ]
        )

    merged["ts_utc"] = pd.to_datetime(merged["ts_utc"], utc=True, errors="coerce")
    merged = merged.dropna(subset=["ts_utc"]).sort_values("ts_utc")

    df = pd.DataFrame()
    df["timestamp"] = merged["ts_utc"]
    df["price"] = merged.get("price")
    df["demand_itsdo"] = merged.get("demand_itsdo")
    df["demand_indo"] = merged.get("demand_indo")
    df["demand_inddem"] = merged.get("demand_inddem")

    return df.reset_index(drop=True)
