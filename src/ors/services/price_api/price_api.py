# bmrs_pipeline_modular.py
# pip install requests pandas numpy python-dateutil

import gc
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from dateutil import parser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# =========================================================
# CONFIGURATION AND MODELS
# =========================================================
@dataclass
class PipelineConfig:
    """Pipeline runtime configuration."""

    base_url: str = "https://data.elexon.co.uk/bmrs/api/v1"
    start_utc: str = "2025-12-01T00:00:00Z"
    end_utc: str = "2026-02-14T00:00:00Z"
    out_dir: str = "bmrs_output_12-2"

    # MID provider filter (optional)
    mid_include_providers: bool = True
    mid_data_providers: list[str] = field(default_factory=lambda: ["APXMIDP"])

    # Output options (CSV only; parquet removed on purpose)
    save_raw_per_endpoint: bool = False
    save_feature_per_endpoint: bool = True
    save_checkpoint_each_endpoint: bool = True
    save_final_csv: bool = True
    return_final_df: bool = False

    # MID extra output options (saved as CSV)
    save_mid_native_long: bool = True
    save_mid_native_wide: bool = True
    save_mid_15m_wide: bool = True

    # Performance / fetching behaviour
    request_timeout: int = 60
    discovery_chunks: int = 2
    max_combos_per_chunk: int = 2
    enable_freq: bool = False


@dataclass(frozen=True)
class EndpointSpec:
    """Specification for one BMRS endpoint family."""

    name: str
    paths: list[str]
    mode: str
    chunk_days: int
    param_styles: list[str]


TIME_CANDIDATES = [
    "settlementDateTime",
    "startTime",
    "endTime",
    "timestamp",
    "dateTime",
    "time",
    "publishTime",
    "publishDateTime",
    "eventStartTime",
    "createdDateTime",
]


# =========================================================
# ENDPOINT SPECS
# =========================================================
def build_endpoint_specs(enable_freq: bool = False) -> dict[str, EndpointSpec]:
    """Create endpoint specs used by the pipeline."""
    endpoints = {
        "price_mid": EndpointSpec(
            name="price_mid",
            paths=["/balancing/pricing/market-index", "/datasets/MID"],
            mode="half_hour",
            chunk_days=7,
            param_styles=["from_to"],  # MID supports from/to in this pipeline
        ),
        "demand_itsdo": EndpointSpec(
            name="demand_itsdo",
            paths=["/datasets/ITSDO"],
            mode="half_hour",
            chunk_days=1,
            param_styles=["publish_window", "from_to", "settlement_window"],
        ),
        "demand_indo": EndpointSpec(
            name="demand_indo",
            paths=["/demand/outturn/stream", "/datasets/INDO"],
            mode="half_hour",
            chunk_days=1,
            param_styles=["settlement_window", "settlement_single", "publish_window", "from_to"],
        ),
        "demand_inddem": EndpointSpec(
            name="demand_inddem",
            paths=["/datasets/INDDEM"],
            mode="half_hour",
            chunk_days=1,
            param_styles=["from_to", "publish_window", "settlement_window"],
        ),
    }

    if enable_freq:
        endpoints["freq"] = EndpointSpec(
            name="freq",
            paths=[
                "/datasets/FREQ/stream",
                "/datasets/FREQ",
                "/system/frequency",
                "/system/frequency/stream",
            ],
            mode="freq_2m",
            chunk_days=1,
            param_styles=["from_to", "publish_window", "settlement_window", "settlement_single"],
        )

    return endpoints


# =========================================================
# HTTP SESSION
# =========================================================
def make_session() -> requests.Session:
    """Create a reusable HTTP session with retry strategy."""
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


# =========================================================
# DATE / JSON HELPERS
# =========================================================
def to_dt(s: str) -> datetime:
    """Parse an ISO-8601 string and return UTC datetime."""
    return parser.isoparse(s).astimezone(timezone.utc)


def to_isoz(dt: datetime) -> str:
    """Format datetime as full ISO string ending with Z."""
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def to_rfc3339_minute_z(dt: datetime) -> str:
    """Format datetime as RFC3339 minute precision: YYYY-MM-DDTHH:MMZ."""
    dt_utc = dt.astimezone(timezone.utc).replace(second=0, microsecond=0)
    return dt_utc.strftime("%Y-%m-%dT%H:%MZ")


def normalize_payload(payload) -> pd.DataFrame:
    """Normalize BMRS payload into a pandas DataFrame.

    Supported payload styles:
    - list
    - dict with one of: data, results, result, items, records
    - generic dict fallback
    """
    if isinstance(payload, list):
        return pd.json_normalize(payload)

    if isinstance(payload, dict):
        for k in ("data", "results", "result", "items", "records"):
            v = payload.get(k)
            if isinstance(v, list):
                return pd.json_normalize(v)

        for v in payload.values():
            if isinstance(v, list):
                return pd.json_normalize(v)

        return pd.json_normalize([payload])

    return pd.DataFrame()


# =========================================================
# PARAMETER BUILDERS
# =========================================================
def base_params(is_stream: bool) -> dict:
    """Return shared params according to endpoint type.

    Non-stream endpoints request JSON explicitly.
    """
    return {} if is_stream else {"format": "json"}


def params_from_to(t0: datetime, t1: datetime, is_stream: bool) -> dict:
    """Build from/to query params."""
    p = base_params(is_stream)
    p.update({"from": to_rfc3339_minute_z(t0), "to": to_rfc3339_minute_z(t1)})
    return p


def params_publish_window(t0: datetime, t1: datetime, is_stream: bool) -> dict:
    """Build publishDateTimeFrom/To query params."""
    p = base_params(is_stream)
    p.update({"publishDateTimeFrom": to_isoz(t0), "publishDateTimeTo": to_isoz(t1)})
    return p


def params_settlement_window(t0: datetime, t1: datetime, is_stream: bool) -> dict:
    """Build settlementDateFrom/To query params."""
    p = base_params(is_stream)
    p.update(
        {
            "settlementDateFrom": t0.strftime("%Y-%m-%d"),
            "settlementDateTo": t1.strftime("%Y-%m-%d"),
        }
    )
    return p


def params_settlement_single(t0: datetime, _t1: datetime, is_stream: bool) -> dict:
    """Build single settlementDate query param."""
    p = base_params(is_stream)
    p.update({"settlementDate": t0.strftime("%Y-%m-%d")})
    return p


def params_price_mid(t0: datetime, t1: datetime, is_stream: bool, cfg: PipelineConfig) -> dict:
    """Build MID params using from/to and optional provider filter."""
    p = params_from_to(t0, t1, is_stream)
    if cfg.mid_include_providers and cfg.mid_data_providers:
        p["dataProviders"] = cfg.mid_data_providers
    return p


PARAM_BUILDERS: dict[str, Callable[[datetime, datetime, bool], dict]] = {
    "from_to": params_from_to,
    "publish_window": params_publish_window,
    "settlement_window": params_settlement_window,
    "settlement_single": params_settlement_single,
}


# =========================================================
# HTTP REQUESTS (ONE FUNCTION PER ENDPOINT FAMILY)
# =========================================================
def request_with_params(
    session: requests.Session,
    base_url: str,
    path: str,
    params: dict,
    timeout: int,
) -> pd.DataFrame:
    """Send one HTTP GET request and normalize response to DataFrame."""
    url = f"{base_url}{path}"
    r = session.get(url, params=params, timeout=timeout)
    if r.status_code >= 400:
        raise requests.HTTPError(f"{r.status_code} | {url} | {r.text[:300]}")
    return normalize_payload(r.json())


def request_price_mid_chunk(
    session: requests.Session,
    cfg: PipelineConfig,
    path: str,
    style: str,
    t0: datetime,
    t1: datetime,
) -> tuple[pd.DataFrame, dict]:
    """Request one time chunk for price_mid."""
    _ = style  # MID is forced to from/to in this pipeline
    params = params_price_mid(t0, t1, path.endswith("/stream"), cfg)
    df = request_with_params(session, cfg.base_url, path, params, cfg.request_timeout)
    return df, params


def request_demand_itsdo_chunk(
    session: requests.Session,
    cfg: PipelineConfig,
    path: str,
    style: str,
    t0: datetime,
    t1: datetime,
) -> tuple[pd.DataFrame, dict]:
    """Request one time chunk for demand_itsdo."""
    builder = PARAM_BUILDERS.get(style, params_from_to)
    params = builder(t0, t1, path.endswith("/stream"))
    df = request_with_params(session, cfg.base_url, path, params, cfg.request_timeout)
    return df, params


def request_demand_indo_chunk(
    session: requests.Session,
    cfg: PipelineConfig,
    path: str,
    style: str,
    t0: datetime,
    t1: datetime,
) -> tuple[pd.DataFrame, dict]:
    """Request one time chunk for demand_indo."""
    builder = PARAM_BUILDERS.get(style, params_from_to)
    params = builder(t0, t1, path.endswith("/stream"))
    df = request_with_params(session, cfg.base_url, path, params, cfg.request_timeout)
    return df, params


def request_demand_inddem_chunk(
    session: requests.Session,
    cfg: PipelineConfig,
    path: str,
    style: str,
    t0: datetime,
    t1: datetime,
) -> tuple[pd.DataFrame, dict]:
    """Request one time chunk for demand_inddem."""
    builder = PARAM_BUILDERS.get(style, params_from_to)
    params = builder(t0, t1, path.endswith("/stream"))
    df = request_with_params(session, cfg.base_url, path, params, cfg.request_timeout)
    return df, params


def request_freq_chunk(
    session: requests.Session,
    cfg: PipelineConfig,
    path: str,
    style: str,
    t0: datetime,
    t1: datetime,
) -> tuple[pd.DataFrame, dict]:
    """Request one time chunk for frequency endpoints."""
    builder = PARAM_BUILDERS.get(style, params_from_to)
    params = builder(t0, t1, path.endswith("/stream"))
    df = request_with_params(session, cfg.base_url, path, params, cfg.request_timeout)
    return df, params


REQUEST_FN_BY_ENDPOINT: dict[str, Callable] = {
    "price_mid": request_price_mid_chunk,
    "demand_itsdo": request_demand_itsdo_chunk,
    "demand_indo": request_demand_indo_chunk,
    "demand_inddem": request_demand_inddem_chunk,
    "freq": request_freq_chunk,
}


# =========================================================
# SCORING / STRATEGY SELECTION
# =========================================================
def _best_time_series_for_score(df: pd.DataFrame) -> pd.Series:
    """Extract the best available time series candidate for scoring rows."""
    for c in TIME_CANDIDATES:
        if c in df.columns:
            ts = pd.to_datetime(df[c], utc=True, errors="coerce")
            if ts.notna().any():
                return ts

    if {"settlementDate", "settlementPeriod"}.issubset(df.columns):
        d = pd.to_datetime(df["settlementDate"], errors="coerce")
        sp = pd.to_numeric(df["settlementPeriod"], errors="coerce")
        local = d.dt.tz_localize(
            "Europe/London", ambiguous="NaT", nonexistent="shift_forward"
        ) + pd.to_timedelta((sp - 1) * 30, unit="m")
        return local.dt.tz_convert("UTC")

    return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")


def score_df(df: pd.DataFrame) -> int:
    """Score candidate DataFrame by size and temporal uniqueness."""
    if df.empty:
        return -1
    ts = _best_time_series_for_score(df)
    nunq = ts.nunique(dropna=True)
    return int(len(df) + 2 * nunq)


# =========================================================
# ROBUST FETCH FOR ONE ENDPOINT SPEC
# =========================================================
def fetch_endpoint_robust(
    session: requests.Session,
    cfg: PipelineConfig,
    spec: EndpointSpec,
    best_strategy_cache: dict[str, tuple[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch data robustly for one endpoint using path/style strategy discovery.

    For first chunks it explores multiple (path, style) combinations.
    Then it reuses the best combination from cache.
    """
    s, e = to_dt(cfg.start_utc), to_dt(cfg.end_utc)
    combos: list[tuple[str, str]] = [(p, st) for p in spec.paths for st in spec.param_styles]
    request_fn = REQUEST_FN_BY_ENDPOINT.get(spec.name, request_demand_itsdo_chunk)

    frames: list[pd.DataFrame] = []
    chunk_logs: list[dict] = []

    cur = s
    chunk_i = 0

    while cur < e:
        nxt = min(cur + timedelta(days=spec.chunk_days), e)

        best_df = pd.DataFrame()
        best_meta = None
        best_score = -1
        best_params = None
        best_err = None

        ordered = combos[:]
        if spec.name in best_strategy_cache and best_strategy_cache[spec.name] in combos:
            cached = best_strategy_cache[spec.name]
            ordered = [cached] + [c for c in combos if c != cached]

        discovery = (spec.name not in best_strategy_cache) and (chunk_i < cfg.discovery_chunks)
        tried = 0

        for path, style in ordered:
            if (not discovery) and (tried >= cfg.max_combos_per_chunk):
                break
            tried += 1

            try:
                df, params_used = request_fn(session, cfg, path, style, cur, nxt)
            except Exception as ex:  # noqa: BLE001
                best_err = str(ex)
                continue

            if df.empty:
                continue

            if discovery:
                sc = score_df(df)
                if sc > best_score:
                    best_score = sc
                    best_df = df
                    best_meta = (path, style, len(df))
                    best_params = dict(params_used)
            else:
                best_df = df
                best_meta = (path, style, len(df))
                best_params = dict(params_used)
                break

        if best_meta and (spec.name not in best_strategy_cache):
            best_strategy_cache[spec.name] = (best_meta[0], best_meta[1])

        if not best_df.empty and best_meta:
            path_used, style_used, _ = best_meta
            best_df["__endpoint_name"] = spec.name
            best_df["__path"] = path_used
            best_df["__style"] = style_used
            best_df["__chunk_start"] = to_isoz(cur)
            best_df["__chunk_end"] = to_isoz(nxt)
            frames.append(best_df)

        chunk_logs.append(
            {
                "endpoint": spec.name,
                "chunk_start": to_isoz(cur),
                "chunk_end": to_isoz(nxt),
                "best_path": best_meta[0] if best_meta else None,
                "best_style": best_meta[1] if best_meta else None,
                "rows": int(best_meta[2]) if best_meta else 0,
                "tried": tried,
                "cached_strategy": str(best_strategy_cache.get(spec.name)),
                "params_used": str(best_params) if best_params else None,
                "last_error": best_err,
            }
        )

        print(
            f"[{spec.name}] {cur:%Y-%m-%d} -> {nxt:%Y-%m-%d} | "
            f"path={chunk_logs[-1]['best_path']} | style={chunk_logs[-1]['best_style']} | "
            f"rows={chunk_logs[-1]['rows']} | tried={tried}"
        )

        cur = nxt
        chunk_i += 1

    if not frames:
        return pd.DataFrame(), pd.DataFrame(chunk_logs)

    raw = pd.concat(frames, ignore_index=True)
    raw = raw.loc[:, ~raw.columns.duplicated()]

    key_candidates = [
        c
        for c in [
            "startTime",
            "settlementDateTime",
            "publishTime",
            "publishDateTime",
            "settlementDate",
            "settlementPeriod",
            "dataProvider",
        ]
        if c in raw.columns
    ]

    if key_candidates:
        subset = key_candidates + (["__path"] if "__path" in raw.columns else [])
        raw = raw.sort_values(key_candidates).drop_duplicates(subset=subset, keep="last")
    else:
        raw = raw.drop_duplicates()

    return raw, pd.DataFrame(chunk_logs)


# =========================================================
# TRANSFORMATIONS
# =========================================================
def attach_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a unified UTC `ts` column from available time fields."""
    if df.empty:
        out = df.copy()
        out["ts"] = pd.to_datetime(pd.Series(dtype="datetime64[ns, UTC]"))
        return out

    out = df.copy()
    out = out.loc[:, ~out.columns.duplicated()]

    for c in TIME_CANDIDATES:
        if c in out.columns:
            ts = pd.to_datetime(out[c], utc=True, errors="coerce")
            if ts.notna().any():
                out["ts"] = ts
                return out

    if {"settlementDate", "settlementPeriod"}.issubset(out.columns):
        d = pd.to_datetime(out["settlementDate"], errors="coerce")
        sp = pd.to_numeric(out["settlementPeriod"], errors="coerce")
        local = d.dt.tz_localize(
            "Europe/London", ambiguous="NaT", nonexistent="shift_forward"
        ) + pd.to_timedelta((sp - 1) * 30, unit="m")
        out["ts"] = local.dt.tz_convert("UTC")
        return out

    out["ts"] = pd.NaT
    return out


def numeric_ts_frame(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Create endpoint features indexed by ts from numeric columns only."""
    if df.empty:
        return pd.DataFrame()

    w = attach_ts(df)
    w["ts"] = pd.to_datetime(w["ts"], utc=True, errors="coerce")
    w = w.dropna(subset=["ts"])
    if w.empty:
        return pd.DataFrame()

    protected = {"ts", "__endpoint_name", "__path", "__style", "__chunk_start", "__chunk_end"}
    cols = [c for c in w.columns if c not in protected]
    if not cols:
        return pd.DataFrame()

    for c in cols:
        w[c] = pd.to_numeric(w[c], errors="coerce")

    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(w[c]) and w[c].notna().any()]
    if not num_cols:
        return pd.DataFrame()

    w = w[["ts"] + num_cols]
    w = w.groupby("ts", as_index=False).mean(numeric_only=True).sort_values("ts")
    w = w.rename(columns={c: f"{prefix}__{c}" for c in num_cols}).set_index("ts")

    w = w.loc[~w.index.duplicated(keep="last")]
    w = w.loc[:, ~w.columns.duplicated()]
    return w


def mid_feature_frame(df_mid_raw: pd.DataFrame, prefix: str = "price_mid"):
    """Build MID-specific features and native MID tables.

    Returns:
    - features wide frame indexed by ts
    - native long frame
    - native wide frame with ts column
    """
    if df_mid_raw.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    w = attach_ts(df_mid_raw)
    w["ts"] = pd.to_datetime(w["ts"], utc=True, errors="coerce")
    w = w.dropna(subset=["ts"])
    if w.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    keep_candidates = [
        "ts",
        "dataProvider",
        "settlementDate",
        "settlementPeriod",
        "price",
        "volume",
    ]
    for c in ["price", "volume", "settlementPeriod"]:
        if c in w.columns:
            w[c] = pd.to_numeric(w[c], errors="coerce")

    keep = [c for c in keep_candidates if c in w.columns]
    mid_long = w[keep].copy().sort_values("ts")

    dedupe_keys = ["ts"] + (["dataProvider"] if "dataProvider" in mid_long.columns else [])
    mid_long = mid_long.drop_duplicates(subset=dedupe_keys, keep="last")

    if "dataProvider" not in mid_long.columns:
        mid_long["dataProvider"] = "UNKNOWN"

    price_pivot = pd.DataFrame(index=sorted(mid_long["ts"].unique()))
    volume_pivot = pd.DataFrame(index=sorted(mid_long["ts"].unique()))

    if "price" in mid_long.columns:
        price_pivot = mid_long.pivot_table(
            index="ts", columns="dataProvider", values="price", aggfunc="last"
        )

    if "volume" in mid_long.columns:
        volume_pivot = mid_long.pivot_table(
            index="ts", columns="dataProvider", values="volume", aggfunc="last"
        )

    if price_pivot.empty and volume_pivot.empty:
        return pd.DataFrame(), mid_long, pd.DataFrame()

    idx = price_pivot.index.union(volume_pivot.index)
    price_pivot = price_pivot.reindex(idx)
    volume_pivot = volume_pivot.reindex(idx)

    mid_wide_native = pd.DataFrame(index=idx)
    for col in price_pivot.columns:
        mid_wide_native[f"price__{col}"] = price_pivot[col]
    for col in volume_pivot.columns:
        mid_wide_native[f"volume__{col}"] = volume_pivot[col]

    feats = pd.DataFrame(index=idx)

    for col in price_pivot.columns:
        feats[f"{prefix}__price__{col}"] = pd.to_numeric(price_pivot[col], errors="coerce")

    for col in volume_pivot.columns:
        feats[f"{prefix}__volume__{col}"] = pd.to_numeric(volume_pivot[col], errors="coerce")

    if not price_pivot.empty:
        feats[f"{prefix}__price__avg_providers"] = price_pivot.mean(axis=1)

    if not volume_pivot.empty:
        feats[f"{prefix}__volume__sum_providers"] = volume_pivot.sum(axis=1)

    if (not price_pivot.empty) and (not volume_pivot.empty):
        common_cols = [c for c in price_pivot.columns if c in volume_pivot.columns]
        if common_cols:
            px = price_pivot[common_cols]
            vol = volume_pivot[common_cols]
            num = (px * vol).sum(axis=1, min_count=1)
            den = vol.sum(axis=1, min_count=1).replace(0, np.nan)
            feats[f"{prefix}__price__vwap_providers"] = num / den

    feats = feats.sort_index()
    feats = feats[~feats.index.duplicated(keep="last")]
    feats = feats.loc[:, ~feats.columns.duplicated()]

    mid_wide_native = mid_wide_native.sort_index()
    mid_wide_native = mid_wide_native[~mid_wide_native.index.duplicated(keep="last")]
    mid_wide_native = mid_wide_native.loc[:, ~mid_wide_native.columns.duplicated()]

    return feats, mid_long, mid_wide_native.reset_index().rename(columns={"index": "ts"})


def resample_15m(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Resample to 15-minute frequency based on endpoint mode."""
    if df.empty:
        return pd.DataFrame()

    d = df.copy()
    d.index = pd.to_datetime(d.index, utc=True, errors="coerce")
    d = d[~d.index.isna()].sort_index()

    if not d.index.is_unique:
        d = d.groupby(level=0).mean(numeric_only=True)

    d = d.select_dtypes(include=[np.number])
    if d.empty:
        return pd.DataFrame()

    if mode == "half_hour":
        # 30m -> 15m (fill one step)
        return d.resample("15min").ffill(limit=1)

    if mode == "freq_2m":
        # 2m -> 15m aggregate stats
        out = d.resample("15min").agg(["mean", "std", "min", "max"])
        out.columns = [f"{c}_{s}" for c, s in out.columns]
        return out

    return d.resample("15min").mean()


# =========================================================
# FEATURE BUILDERS PER ENDPOINT TYPE
# =========================================================
def build_features_price_mid(raw: pd.DataFrame, spec: EndpointSpec):
    """Build MID features and resample to 15m."""
    feats, mid_long_native, mid_wide_native = mid_feature_frame(raw, prefix=spec.name)
    rs = resample_15m(feats, mode=spec.mode)
    return rs, mid_long_native, mid_wide_native


def build_features_generic(raw: pd.DataFrame, spec: EndpointSpec):
    """Build generic numeric features and resample to 15m."""
    feats = numeric_ts_frame(raw, prefix=spec.name)
    rs = resample_15m(feats, mode=spec.mode)
    return rs, pd.DataFrame(), pd.DataFrame()


# =========================================================
# FINAL POST-PROCESSING
# =========================================================
def choose_target_price_col(columns) -> str | None:
    """Pick the best available target price column by priority."""
    cols = list(columns)
    lower = {c: c.lower() for c in cols}

    priorities = [
        "price_mid__price__vwap_providers",
        "price_mid__price__avg_providers",
        "price_mid__price__n2exmidp",
        "price_mid__price__apxmidp",
        "price_mid__price",
        "marketindexprice",
        "price",
    ]
    for p in priorities:
        for c in cols:
            if p in lower[c]:
                return c

    for c in cols:
        if "price" in lower[c]:
            return c
    return None


def add_calendar_and_lags(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add target, calendar features, and lag features."""
    tgt_col = choose_target_price_col(dataset.columns)
    dataset["target_price"] = dataset[tgt_col] if tgt_col else np.nan

    dataset["hour"] = dataset.index.hour.astype("int16")
    dataset["dayofweek"] = dataset.index.dayofweek.astype("int8")
    dataset["month"] = dataset.index.month.astype("int8")
    dataset["is_weekend"] = (dataset["dayofweek"] >= 5).astype("int8")

    for lag in [1, 2, 4, 96]:
        dataset[f"target_lag_{lag}"] = dataset["target_price"].shift(lag).astype("float32")

    return dataset


# =========================================================
# I/O HELPERS (CSV ONLY)
# =========================================================
def save_csv_if_needed(df: pd.DataFrame, path: str, enabled: bool, index: bool = False):
    """Save DataFrame as CSV if enabled and non-empty."""
    if enabled and not df.empty:
        df.to_csv(path, index=index)
        print(f"[SAVE] {path}")


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to float32 to reduce memory usage."""
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")
    return out


# =========================================================
# MODEL DATASET EXPORT (price + demand subset)
# =========================================================
def extract_price_model_frame(data: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """Create the compact model input frame used for price-demand experiments.

    Output columns:
    - price          <- price_mid__price__APXMIDP
    - demand_itsdo   <- demand_itsdo__demand
    - timestamp      <- ts_utc
    - demand_indo    <- demand_indo__initialDemandOutturn
    - demand_inddem  <- demand_inddem__demand

    If `strict=False`, missing source columns are filled with NaN.
    If `strict=True`, a missing source column raises KeyError.
    """
    column_map = {
        "price": "price_mid__price__APXMIDP",
        "demand_itsdo": "demand_itsdo__demand",
        "timestamp": "ts_utc",
        "demand_indo": "demand_indo__initialDemandOutturn",
        "demand_inddem": "demand_inddem__demand",
    }

    out = pd.DataFrame(index=data.index)
    for out_col, src_col in column_map.items():
        if src_col in data.columns:
            out[out_col] = data[src_col]
        elif strict:
            raise KeyError(f"Required column not found: {src_col}")
        else:
            out[out_col] = np.nan

    return out


def export_price_model_csv_from_merged(
    input_csv_path: str,
    output_csv_path: str,
    strict: bool = False,
) -> pd.DataFrame:
    """Read a merged BMRS CSV and export a compact price model CSV.

    This reproduces the transformation:
    - read merged dataset CSV
    - select/rename relevant columns
    - write `price_data.csv`
    """
    data = pd.read_csv(input_csv_path)
    out = extract_price_model_frame(data, strict=strict)
    out.to_csv(output_csv_path, index=False)
    print(f"[SAVE] {output_csv_path}")
    return out


# =========================================================
# MAIN PIPELINE
# =========================================================
def build_bmrs_dataset_15m_all(cfg: PipelineConfig):
    """Run complete BMRS pipeline and generate output files.

    Notes:
    - Parquet output has intentionally been removed.
    - All generated files are CSV.
    """
    os.makedirs(cfg.out_dir, exist_ok=True)

    endpoints = build_endpoint_specs(cfg.enable_freq)
    session = make_session()
    best_strategy_cache: dict[str, tuple[str, str]] = {}

    idx = pd.date_range(
        start=to_dt(cfg.start_utc),
        end=to_dt(cfg.end_utc),
        freq="15min",
        inclusive="left",
        tz="UTC",
    )
    dataset = pd.DataFrame(index=idx)
    dataset.index.name = "ts_utc"

    logs_all: list[pd.DataFrame] = []

    mid_long_native = pd.DataFrame()
    mid_wide_native = pd.DataFrame()

    try:
        for name, spec in endpoints.items():
            print(f"\n=== Endpoint: {name} ===")

            raw, logs = fetch_endpoint_robust(session, cfg, spec, best_strategy_cache)
            logs_all.append(logs)
            print(f"[OK] {name:<14} raw_rows={len(raw):,}")

            if cfg.save_raw_per_endpoint and not raw.empty:
                raw_path = os.path.join(cfg.out_dir, f"raw_{name}.csv")
                raw.to_csv(raw_path, index=False)
                print(f"[SAVE] {raw_path}")

            if name == "price_mid":
                rs, mid_long_native, mid_wide_native = build_features_price_mid(raw, spec)
            else:
                rs, _, _ = build_features_generic(raw, spec)

            del raw
            gc.collect()

            if rs.empty:
                print(f"[FEAT] {name:<14} -> (0, 0)")
                continue

            rs = rs.reindex(idx)
            rs = downcast_numeric(rs)

            print(f"[FEAT] {name:<14} -> {rs.shape}")

            if cfg.save_feature_per_endpoint:
                feat_path = os.path.join(cfg.out_dir, f"feat_{name}_15m.csv")
                rs.reset_index().to_csv(feat_path, index=False)
                print(f"[SAVE] {feat_path}")

            dataset = dataset.join(rs, how="left")
            dataset = dataset.loc[:, ~dataset.columns.duplicated()]

            del rs
            gc.collect()

            if cfg.save_checkpoint_each_endpoint:
                ckpt_path = os.path.join(cfg.out_dir, "dataset_checkpoint.csv")
                dataset.reset_index().to_csv(ckpt_path, index=False)
                print(f"[CKPT] {ckpt_path} | shape={dataset.shape}")

        # Optional MID outputs (CSV)
        save_csv_if_needed(
            mid_long_native,
            os.path.join(cfg.out_dir, "mid_long_native.csv"),
            cfg.save_mid_native_long,
            index=False,
        )

        save_csv_if_needed(
            mid_wide_native,
            os.path.join(cfg.out_dir, "mid_wide_native.csv"),
            cfg.save_mid_native_wide,
            index=False,
        )

        if not mid_wide_native.empty and cfg.save_mid_15m_wide:
            tmp = mid_wide_native.copy()
            tmp["ts"] = pd.to_datetime(tmp["ts"], utc=True, errors="coerce")
            tmp = tmp.dropna(subset=["ts"]).set_index("ts").sort_index()
            tmp = tmp[~tmp.index.duplicated(keep="last")]
            mid_wide_15m = (
                tmp.reindex(idx).ffill(limit=1).reset_index().rename(columns={"index": "ts_utc"})
            )
            p = os.path.join(cfg.out_dir, "mid_wide_15m.csv")
            mid_wide_15m.to_csv(p, index=False)
            print(f"[SAVE] {p}")

        # Final post-processing
        dataset = add_calendar_and_lags(dataset)
        final = dataset.reset_index()

        # Final output (CSV only)
        out_paths = {}
        if cfg.save_final_csv:
            p = os.path.join(cfg.out_dir, "bmrs_dataset_15m.csv")
            final.to_csv(p, index=False)
            out_paths["csv"] = p

        # Diagnostics
        logs_df = pd.concat(logs_all, ignore_index=True) if logs_all else pd.DataFrame()
        p_diag = os.path.join(cfg.out_dir, "bmrs_request_diagnostics.csv")
        logs_df.to_csv(p_diag, index=False)
        out_paths["diagnostics"] = p_diag

        print("\n✅ ALL-ENDPOINTS pipeline completed")
        print(f"- rows: {len(final):,} | columns: {final.shape[1]}")
        for k, v in out_paths.items():
            print(f"- {k}: {v}")
        print(f"- cached strategies: {best_strategy_cache}")

        target_non_null = (
            int(final["target_price"].notna().sum()) if "target_price" in final.columns else 0
        )
        print(f"[CHECK] target_price non-nulls: {target_non_null}")

        itsdo_cols = [
            c for c in final.columns if c.startswith("demand_itsdo__") and "demand" in c.lower()
        ]
        itsdo_non_null = int(final[itsdo_cols[0]].notna().sum()) if itsdo_cols else 0
        print(f"[CHECK] ITSDO non-nulls (main column): {itsdo_non_null}")

        if cfg.return_final_df:
            return final, out_paths

        del final
        gc.collect()
        return None, out_paths

    finally:
        session.close()


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    cfg = PipelineConfig(
        base_url="https://data.elexon.co.uk/bmrs/api/v1",
        start_utc="2025-06-01T00:00:00Z",
        end_utc="2026-02-14T00:00:00Z",
        out_dir="bmrs_output_12-2",
        mid_include_providers=True,
        mid_data_providers=["APXMIDP"],
        save_raw_per_endpoint=False,
        save_feature_per_endpoint=True,
        save_checkpoint_each_endpoint=True,
        save_final_csv=True,
        return_final_df=False,
        save_mid_native_long=True,
        save_mid_native_wide=True,
        save_mid_15m_wide=True,
        request_timeout=60,
        discovery_chunks=2,
        max_combos_per_chunk=2,
        enable_freq=False,
    )

    _, paths = build_bmrs_dataset_15m_all(cfg)

    print("\nGenerated files:")
    for k, v in paths.items():
        print(f"{k}: {v}")

    # Create compact training CSV (price + demand subset) from final merged CSV.
    merged_csv = paths.get("csv")
    if merged_csv:
        model_csv = os.path.join(cfg.out_dir, "price_data.csv")
        try:
            export_price_model_csv_from_merged(
                input_csv_path=merged_csv,
                output_csv_path=model_csv,
                strict=False,
            )
            print(f"price_data_csv: {model_csv}")
        except Exception as ex:  # noqa: BLE001
            print(f"[WARN] Could not create price_data.csv: {ex}")
