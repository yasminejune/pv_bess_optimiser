"""
Pytest suite for `src/ors/services/price_api.py`.

Includes:
- Pure unit tests (default)
- Optional real API smoke tests (disabled by default)

Enable real API tests with:
    RUN_BMRS_API_TESTS=1 pytest -q
"""

import importlib
import inspect
import os
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
from pandas.api.types import is_object_dtype, is_string_dtype


# -------------------------------------------------------------------
# Import helper
# -------------------------------------------------------------------
def _ensure_src_on_path() -> None:
    """Ensure `src/` is importable when running pytest from project root."""
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd(),
        here.parents[3] if len(here.parents) >= 4 else None,  # tests/unit/services -> repo root
        here.parents[2] if len(here.parents) >= 3 else None,
        here.parents[1] if len(here.parents) >= 2 else None,
    ]
    for root in [c for c in candidates if c is not None]:
        src = root / "src"
        if src.exists():
            s = str(src)
            if s not in sys.path:
                sys.path.insert(0, s)


@pytest.fixture(scope="session")
def bmrs():
    _ensure_src_on_path()
    module_name = os.getenv("BMRS_MODULE", "ors.services.price_api")
    return importlib.import_module(module_name)


@pytest.fixture()
def t0_t1():
    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc)
    return t0, t1


# -------------------------------------------------------------------
# Basic module sanity
# -------------------------------------------------------------------
def test_module_exports_expected_symbols(bmrs):
    expected = [
        "PipelineConfig",
        "EndpointSpec",
        "make_session",
        "to_dt",
        "to_isoz",
        "to_rfc3339_minute_z",
        "normalize_payload",
        "build_endpoint_specs",
        "fetch_endpoint_robust",
        "attach_ts",
        "numeric_ts_frame",
        "mid_feature_frame",
        "resample_15m",
        "build_bmrs_dataset_15m_all",
    ]
    for name in expected:
        assert hasattr(bmrs, name), f"Missing symbol: {name}"


def test_make_session_returns_requests_session(bmrs):
    s = bmrs.make_session()
    try:
        assert isinstance(s, requests.Session)
    finally:
        s.close()


# -------------------------------------------------------------------
# Datetime helpers
# -------------------------------------------------------------------
def test_to_dt_to_isoz_roundtrip(bmrs):
    x = "2026-01-01T00:00:00Z"
    dt = bmrs.to_dt(x)
    assert dt.tzinfo is not None
    assert bmrs.to_isoz(dt) == x


def test_to_rfc3339_minute_z(bmrs):
    dt = datetime(2026, 1, 1, 10, 12, 59, tzinfo=timezone.utc)
    assert bmrs.to_rfc3339_minute_z(dt) == "2026-01-01T10:12Z"


# -------------------------------------------------------------------
# Payload normalization
# -------------------------------------------------------------------
def test_normalize_payload_list(bmrs):
    payload = [{"a": 1}, {"a": 2}]
    df = bmrs.normalize_payload(payload)
    assert list(df["a"]) == [1, 2]


def test_normalize_payload_dict_with_data(bmrs):
    payload = {"data": [{"x": 1}]}
    df = bmrs.normalize_payload(payload)
    assert len(df) == 1
    assert df.loc[0, "x"] == 1


def test_normalize_payload_fallback_values_list(bmrs):
    payload = {"meta": 1, "items": [{"k": "v"}]}
    df = bmrs.normalize_payload(payload)
    assert len(df) == 1
    assert df.loc[0, "k"] == "v"


def test_normalize_payload_flat_dict(bmrs):
    payload = {"k": "v"}
    df = bmrs.normalize_payload(payload)
    assert len(df) == 1
    assert df.loc[0, "k"] == "v"


def test_normalize_payload_unknown_type_returns_empty(bmrs):
    df = bmrs.normalize_payload("not_json")
    assert df.empty


# -------------------------------------------------------------------
# Endpoint specs
# -------------------------------------------------------------------
def test_build_endpoint_specs_default_without_freq(bmrs):
    eps = bmrs.build_endpoint_specs(enable_freq=False)
    assert "price_mid" in eps
    assert "demand_itsdo" in eps
    assert "demand_indo" in eps
    assert "demand_inddem" in eps
    assert "freq" not in eps


def test_build_endpoint_specs_with_freq(bmrs):
    eps = bmrs.build_endpoint_specs(enable_freq=True)
    assert "freq" in eps
    assert eps["freq"].mode == "freq_2m"


# -------------------------------------------------------------------
# Param builders
# -------------------------------------------------------------------
def test_base_params_stream_and_non_stream(bmrs):
    assert bmrs.base_params(is_stream=False) == {"format": "json"}
    assert bmrs.base_params(is_stream=True) == {}


def test_params_from_to(bmrs, t0_t1):
    t0, t1 = t0_t1
    p = bmrs.params_from_to(t0, t1, is_stream=False)
    assert p["format"] == "json"
    assert p["from"].endswith("Z") and p["to"].endswith("Z")
    assert "publishDateTimeFrom" not in p


def test_params_publish_window(bmrs, t0_t1):
    t0, t1 = t0_t1
    p = bmrs.params_publish_window(t0, t1, is_stream=False)
    assert p["format"] == "json"
    assert "publishDateTimeFrom" in p and "publishDateTimeTo" in p


def test_params_settlement_window(bmrs, t0_t1):
    t0, t1 = t0_t1
    p = bmrs.params_settlement_window(t0, t1, is_stream=False)
    assert p["settlementDateFrom"] == "2026-01-01"
    assert p["settlementDateTo"] == "2026-01-01"


def test_params_settlement_single(bmrs, t0_t1):
    t0, _ = t0_t1
    p = bmrs.params_settlement_single(t0, t0, is_stream=False)
    assert p["settlementDate"] == "2026-01-01"


def test_params_price_mid_with_provider_filter(bmrs, t0_t1):
    t0, t1 = t0_t1
    cfg = bmrs.PipelineConfig(
        mid_include_providers=True,
        mid_data_providers=["APXMIDP"],
    )
    p = bmrs.params_price_mid(t0, t1, is_stream=False, cfg=cfg)
    assert p["format"] == "json"
    assert p["dataProviders"] == ["APXMIDP"]
    assert "from" in p and "to" in p


def test_params_price_mid_without_provider_filter(bmrs, t0_t1):
    t0, t1 = t0_t1
    cfg = bmrs.PipelineConfig(
        mid_include_providers=False,
        mid_data_providers=["APXMIDP"],
    )
    p = bmrs.params_price_mid(t0, t1, is_stream=True, cfg=cfg)
    assert "format" not in p  # stream => no format
    assert "dataProviders" not in p


# -------------------------------------------------------------------
# HTTP request wrapper and endpoint request fns
# -------------------------------------------------------------------
def test_request_with_params_success(monkeypatch, bmrs):
    class FakeResp:
        status_code = 200

        @staticmethod
        def json():
            return {"data": [{"ok": 1}]}

    class FakeSession:
        def get(self, url, params=None, timeout=60):
            return FakeResp()

    df = bmrs.request_with_params(
        session=FakeSession(),
        base_url="https://example.org",
        path="/x",
        params={"a": 1},
        timeout=10,
    )
    assert len(df) == 1
    assert df.loc[0, "ok"] == 1


def test_request_with_params_http_error(monkeypatch, bmrs):
    class FakeResp:
        status_code = 500
        text = "boom"

        @staticmethod
        def json():
            return {"error": "boom"}

    class FakeSession:
        def get(self, url, params=None, timeout=60):
            return FakeResp()

    with pytest.raises(requests.HTTPError):
        bmrs.request_with_params(
            session=FakeSession(),
            base_url="https://example.org",
            path="/x",
            params={},
            timeout=10,
        )


@pytest.mark.parametrize(
    "fn_name,endpoint_name,style",
    [
        ("request_price_mid_chunk", "price_mid", "from_to"),
        ("request_demand_itsdo_chunk", "demand_itsdo", "publish_window"),
        ("request_demand_indo_chunk", "demand_indo", "settlement_window"),
        ("request_demand_inddem_chunk", "demand_inddem", "settlement_single"),
        ("request_freq_chunk", "freq", "from_to"),
    ],
)
def test_request_chunk_functions_build_params(
    monkeypatch, bmrs, t0_t1, fn_name, endpoint_name, style
):
    t0, t1 = t0_t1
    called = {}

    def fake_request_with_params(session, base_url, path, params, timeout):
        called["params"] = params
        called["path"] = path
        return pd.DataFrame([{"value": 1}])

    monkeypatch.setattr(bmrs, "request_with_params", fake_request_with_params)

    fn = getattr(bmrs, fn_name)
    cfg = bmrs.PipelineConfig()
    df, params = fn(
        session=object(),
        cfg=cfg,
        path="/datasets/MID" if endpoint_name == "price_mid" else "/datasets/ITSDO",
        style=style,
        t0=t0,
        t1=t1,
    )

    assert isinstance(df, pd.DataFrame)
    assert isinstance(params, dict)
    assert "params" in called

    if endpoint_name == "price_mid":
        # Price MID always uses from/to, independently of style
        assert "from" in params and "to" in params
    else:
        # Non-MID depends on selected style builder
        assert len(params) > 0


# -------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------
def test_best_time_series_for_score_from_time_column(bmrs):
    df = pd.DataFrame({"startTime": ["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"]})
    ts = bmrs._best_time_series_for_score(df)
    assert ts.notna().all()
    assert "datetime64" in str(ts.dtype) and "UTC" in str(ts.dtype)


def test_best_time_series_for_score_from_settlement_cols(bmrs):
    df = pd.DataFrame(
        {
            "settlementDate": ["2026-01-01", "2026-01-01"],
            "settlementPeriod": [1, 2],
        }
    )
    ts = bmrs._best_time_series_for_score(df)
    assert ts.notna().all()


def test_score_df_empty_and_non_empty(bmrs):
    assert bmrs.score_df(pd.DataFrame()) == -1
    df = pd.DataFrame({"startTime": ["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"]})
    assert bmrs.score_df(df) > 0


# -------------------------------------------------------------------
# Robust fetch
# -------------------------------------------------------------------
def test_fetch_endpoint_robust_discovery_and_cache(monkeypatch, bmrs):
    cfg = bmrs.PipelineConfig(
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-02T00:00:00Z",
        discovery_chunks=2,
        max_combos_per_chunk=5,
    )
    spec = bmrs.EndpointSpec(
        name="dummy",
        paths=["/a"],
        mode="half_hour",
        chunk_days=1,
        param_styles=["from_to", "publish_window"],
    )
    cache = {}

    # publish_window returns better score (2 rows instead of 1)
    def fake_request_fn(session, cfg_, path, style, t0, t1):
        if style == "publish_window":
            df = pd.DataFrame(
                {"startTime": ["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"], "v": [1, 2]}
            )
        else:
            df = pd.DataFrame({"startTime": ["2026-01-01T00:00:00Z"], "v": [3]})
        return df, {"style": style}

    monkeypatch.setitem(bmrs.REQUEST_FN_BY_ENDPOINT, "dummy", fake_request_fn)

    raw, logs = bmrs.fetch_endpoint_robust(
        session=object(), cfg=cfg, spec=spec, best_strategy_cache=cache
    )

    assert not raw.empty
    assert not logs.empty
    assert cache["dummy"][1] == "publish_window"
    assert "__endpoint_name" in raw.columns
    assert "__path" in raw.columns
    assert "__style" in raw.columns


def test_fetch_endpoint_robust_when_all_calls_fail(monkeypatch, bmrs):
    cfg = bmrs.PipelineConfig(
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-02T00:00:00Z",
        discovery_chunks=1,
        max_combos_per_chunk=2,
    )
    spec = bmrs.EndpointSpec(
        name="always_fail",
        paths=["/x"],
        mode="half_hour",
        chunk_days=1,
        param_styles=["from_to"],
    )
    cache = {}

    def fake_request_fn(session, cfg_, path, style, t0, t1):
        raise RuntimeError("boom")

    monkeypatch.setitem(bmrs.REQUEST_FN_BY_ENDPOINT, "always_fail", fake_request_fn)

    raw, logs = bmrs.fetch_endpoint_robust(
        session=object(), cfg=cfg, spec=spec, best_strategy_cache=cache
    )

    assert raw.empty
    assert len(logs) == 1
    assert logs.loc[0, "rows"] == 0
    assert "boom" in str(logs.loc[0, "last_error"])


# -------------------------------------------------------------------
# Transformations
# -------------------------------------------------------------------
def test_attach_ts_from_time_column(bmrs):
    df = pd.DataFrame({"startTime": ["2026-01-01T00:00:00Z"]})
    out = bmrs.attach_ts(df)
    assert "ts" in out.columns
    assert pd.notna(out.loc[0, "ts"])


def test_attach_ts_from_settlement_pair(bmrs):
    df = pd.DataFrame({"settlementDate": ["2026-01-01"], "settlementPeriod": [1]})
    out = bmrs.attach_ts(df)
    assert "ts" in out.columns
    assert pd.notna(out.loc[0, "ts"])


def test_attach_ts_without_time_columns_returns_nat(bmrs):
    df = pd.DataFrame({"x": [1]})
    out = bmrs.attach_ts(df)
    assert "ts" in out.columns
    assert out["ts"].isna().all()


def test_numeric_ts_frame_ok_and_empty_cases(bmrs):
    good = pd.DataFrame(
        {
            "startTime": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "a": ["1", "3"],
            "b": [2, 4],
            "txt": ["x", "y"],
        }
    )
    out = bmrs.numeric_ts_frame(good, prefix="ep")
    assert not out.empty
    assert "ep__a" in out.columns and "ep__b" in out.columns

    bad = pd.DataFrame({"startTime": ["2026-01-01T00:00:00Z"], "txt": ["x"]})
    out_bad = bmrs.numeric_ts_frame(bad, prefix="ep")
    assert out_bad.empty


def test_mid_feature_frame_with_provider_price_and_volume(bmrs):
    raw = pd.DataFrame(
        {
            "startTime": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:30:00Z",
            ],
            "dataProvider": ["APXMIDP", "N2EXMIDP", "APXMIDP"],
            "price": [100, 110, 120],
            "volume": [10, 20, 5],
            "settlementDate": ["2026-01-01", "2026-01-01", "2026-01-01"],
            "settlementPeriod": [1, 1, 2],
        }
    )
    feats, long_df, wide_native = bmrs.mid_feature_frame(raw, prefix="price_mid")
    assert not feats.empty
    assert not long_df.empty
    assert not wide_native.empty
    assert "price_mid__price__avg_providers" in feats.columns
    assert "price_mid__price__vwap_providers" in feats.columns


def test_mid_feature_frame_without_data_provider_column(bmrs):
    raw = pd.DataFrame(
        {
            "startTime": ["2026-01-01T00:00:00Z"],
            "price": [100],
            "volume": [10],
        }
    )
    feats, long_df, wide_native = bmrs.mid_feature_frame(raw, prefix="price_mid")
    assert not long_df.empty
    assert "dataProvider" in long_df.columns
    assert "UNKNOWN" in set(long_df["dataProvider"].astype(str))


def test_resample_15m_modes(bmrs):
    idx = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"], utc=True)
    df = pd.DataFrame({"x": [1.0, 3.0]}, index=idx)

    hh = bmrs.resample_15m(df, mode="half_hour")
    assert len(hh) >= 3  # 00:00, 00:15, 00:30

    # freq_2m aggregated columns with suffix
    idx2 = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="2min", tz="UTC")
    df2 = pd.DataFrame({"f": range(8)}, index=idx2)
    fq = bmrs.resample_15m(df2, mode="freq_2m")
    assert any(col.endswith("_mean") for col in fq.columns)

    other = bmrs.resample_15m(df, mode="unknown")
    assert not other.empty


def test_build_features_wrappers(bmrs):
    spec_mid = bmrs.EndpointSpec(
        name="price_mid",
        paths=["/datasets/MID"],
        mode="half_hour",
        chunk_days=1,
        param_styles=["from_to"],
    )
    raw_mid = pd.DataFrame(
        {
            "startTime": ["2026-01-01T00:00:00Z"],
            "dataProvider": ["APXMIDP"],
            "price": [100],
            "volume": [10],
        }
    )
    rs_mid, mid_long, mid_wide = bmrs.build_features_price_mid(raw_mid, spec_mid)
    assert isinstance(rs_mid, pd.DataFrame)
    assert isinstance(mid_long, pd.DataFrame)
    assert isinstance(mid_wide, pd.DataFrame)

    spec_gen = bmrs.EndpointSpec(
        name="demand_itsdo",
        paths=["/datasets/ITSDO"],
        mode="half_hour",
        chunk_days=1,
        param_styles=["from_to"],
    )
    raw_gen = pd.DataFrame({"startTime": ["2026-01-01T00:00:00Z"], "demand": [1000]})
    rs_gen, _, _ = bmrs.build_features_generic(raw_gen, spec_gen)
    assert not rs_gen.empty


# -------------------------------------------------------------------
# Final post-processing
# -------------------------------------------------------------------
def test_choose_target_price_col_priority_and_fallback(bmrs):
    cols = [
        "foo",
        "price_mid__price__avg_providers",
        "something_price_else",
    ]
    assert bmrs.choose_target_price_col(cols) == "price_mid__price__avg_providers"

    cols2 = ["abc", "myPriceCol"]
    assert bmrs.choose_target_price_col(cols2) == "myPriceCol"

    assert bmrs.choose_target_price_col(["a", "b"]) is None


def test_add_calendar_and_lags(bmrs):
    idx = pd.date_range("2026-01-01T00:00:00Z", periods=120, freq="15min", tz="UTC")
    df = pd.DataFrame({"price_mid__price__avg_providers": range(120)}, index=idx)
    out = bmrs.add_calendar_and_lags(df.copy())

    for c in [
        "target_price",
        "hour",
        "dayofweek",
        "month",
        "is_weekend",
        "target_lag_1",
        "target_lag_96",
    ]:
        assert c in out.columns
    assert pd.isna(out["target_lag_1"].iloc[0])


# -------------------------------------------------------------------
# IO helpers
# -------------------------------------------------------------------
def test_save_csv_if_needed_and_downcast(tmp_path, bmrs):
    p = tmp_path / "x.csv"
    df = pd.DataFrame({"a": [1.0], "txt": ["ok"]})

    bmrs.save_csv_if_needed(df, str(p), enabled=True, index=False)
    assert p.exists()

    p2 = tmp_path / "y.csv"
    bmrs.save_csv_if_needed(df, str(p2), enabled=False, index=False)
    assert not p2.exists()

    out = bmrs.downcast_numeric(df)
    assert str(out["a"].dtype) == "float32"
    # non-numeric dtype may be object or pandas StringDtype depending on configuration
    assert is_string_dtype(out["txt"]) or is_object_dtype(out["txt"])


# -------------------------------------------------------------------
# Main pipeline (fast unit style with monkeypatch)
# -------------------------------------------------------------------
def test_build_pipeline_quick_mid_only(tmp_path, monkeypatch, bmrs):
    # Keep only one endpoint for a deterministic fast test
    spec_mid = bmrs.EndpointSpec(
        name="price_mid",
        paths=["/datasets/MID"],
        mode="half_hour",
        chunk_days=1,
        param_styles=["from_to"],
    )
    monkeypatch.setattr(bmrs, "build_endpoint_specs", lambda enable_freq: {"price_mid": spec_mid})

    class FakeSession:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    fake_session = FakeSession()
    monkeypatch.setattr(bmrs, "make_session", lambda: fake_session)

    def fake_fetch_endpoint_robust(session, cfg, spec, best_strategy_cache):
        raw = pd.DataFrame(
            {
                "startTime": ["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"],
                "dataProvider": ["APXMIDP", "APXMIDP"],
                "price": [100.0, 102.0],
                "volume": [10.0, 12.0],
            }
        )
        logs = pd.DataFrame(
            [
                {
                    "endpoint": spec.name,
                    "chunk_start": cfg.start_utc,
                    "chunk_end": cfg.end_utc,
                    "best_path": "/datasets/MID",
                    "best_style": "from_to",
                    "rows": 2,
                    "tried": 1,
                    "cached_strategy": "('/datasets/MID', 'from_to')",
                    "params_used": "{'from':'x','to':'y'}",
                    "last_error": None,
                }
            ]
        )
        return raw, logs

    monkeypatch.setattr(bmrs, "fetch_endpoint_robust", fake_fetch_endpoint_robust)

    cfg = bmrs.PipelineConfig(
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-01T01:00:00Z",
        out_dir=str(tmp_path),
        save_raw_per_endpoint=False,
        save_feature_per_endpoint=False,
        save_checkpoint_each_endpoint=False,
        save_final_csv=True,
        return_final_df=True,
        save_mid_native_long=False,
        save_mid_native_wide=False,
        save_mid_15m_wide=False,
    )

    final, out_paths = bmrs.build_bmrs_dataset_15m_all(cfg)
    assert fake_session.closed is True
    assert isinstance(final, pd.DataFrame)
    assert len(final) == 4  # 1 hour in 15m bins
    assert "target_price" in final.columns
    assert "csv" in out_paths and Path(out_paths["csv"]).exists()
    assert "diagnostics" in out_paths and Path(out_paths["diagnostics"]).exists()


def test_build_pipeline_closes_session_on_exception(tmp_path, monkeypatch, bmrs):
    spec_mid = bmrs.EndpointSpec(
        name="price_mid",
        paths=["/datasets/MID"],
        mode="half_hour",
        chunk_days=1,
        param_styles=["from_to"],
    )
    monkeypatch.setattr(bmrs, "build_endpoint_specs", lambda enable_freq: {"price_mid": spec_mid})

    class FakeSession:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    fake_session = FakeSession()
    monkeypatch.setattr(bmrs, "make_session", lambda: fake_session)

    def fail_fetch(*args, **kwargs):
        raise RuntimeError("forced-error")

    monkeypatch.setattr(bmrs, "fetch_endpoint_robust", fail_fetch)

    cfg = bmrs.PipelineConfig(
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-01T01:00:00Z",
        out_dir=str(tmp_path),
    )

    with pytest.raises(RuntimeError, match="forced-error"):
        bmrs.build_bmrs_dataset_15m_all(cfg)

    assert fake_session.closed is True


# -------------------------------------------------------------------
# Optional real API smoke tests
# -------------------------------------------------------------------
RUN_BMRS_API_TESTS = os.getenv("RUN_BMRS_API_TESTS", "0") == "1"


def _api_cfg(bmrs):
    return bmrs.PipelineConfig(
        start_utc="2026-02-01T00:00:00Z",
        end_utc="2026-02-01T01:00:00Z",
        request_timeout=30,
        save_raw_per_endpoint=False,
        save_feature_per_endpoint=False,
        save_checkpoint_each_endpoint=False,
        save_final_csv=False,
        return_final_df=False,
        save_mid_native_long=False,
        save_mid_native_wide=False,
        save_mid_15m_wide=False,
    )


@pytest.mark.skipif(not RUN_BMRS_API_TESTS, reason="RUN_BMRS_API_TESTS is disabled.")
@pytest.mark.parametrize(
    "fn_name,path,style",
    [
        ("request_price_mid_chunk", "/datasets/MID", "from_to"),
        ("request_demand_itsdo_chunk", "/datasets/ITSDO", "from_to"),
        ("request_demand_indo_chunk", "/datasets/INDO", "from_to"),
        ("request_demand_inddem_chunk", "/datasets/INDDEM", "from_to"),
    ],
)
def test_real_api_request_functions_smoke(bmrs, fn_name, path, style):
    session = bmrs.make_session()
    try:
        cfg = _api_cfg(bmrs)
        fn: Callable = getattr(bmrs, fn_name)
        t0 = bmrs.to_dt(cfg.start_utc)
        t1 = bmrs.to_dt(cfg.end_utc)
        df, params = fn(session, cfg, path, style, t0, t1)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(params, dict)
    finally:
        session.close()


@pytest.mark.skipif(not RUN_BMRS_API_TESTS, reason="RUN_BMRS_API_TESTS is disabled.")
def test_real_api_fetch_endpoint_robust_smoke(bmrs):
    cfg = _api_cfg(bmrs)
    spec = bmrs.EndpointSpec(
        name="price_mid",
        paths=["/datasets/MID"],
        mode="half_hour",
        chunk_days=1,
        param_styles=["from_to"],
    )
    session = bmrs.make_session()
    try:
        raw, logs = bmrs.fetch_endpoint_robust(
            session=session,
            cfg=cfg,
            spec=spec,
            best_strategy_cache={},
        )
        assert isinstance(raw, pd.DataFrame)
        assert isinstance(logs, pd.DataFrame)
        assert len(logs) >= 1
    finally:
        session.close()


# =========================
# Extra tests for branch coverage
# =========================


def test_cov_normalize_payload_dict_unknown_list_key(bmrs):
    payload = {"random_list": [{"a": 1}, {"a": 2}]}
    df = bmrs.normalize_payload(payload)
    assert len(df) == 2
    assert "a" in df.columns


def test_cov_best_time_series_skips_invalid_first_candidate_then_uses_second(bmrs):
    df = pd.DataFrame(
        {
            "startTime": ["not-a-date", "not-a-date"],
            "settlementDateTime": ["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"],
        }
    )
    ts = bmrs._best_time_series_for_score(df)
    assert ts.notna().all()


def test_cov_best_time_series_returns_nat_when_no_time_columns(bmrs):
    ts = bmrs._best_time_series_for_score(pd.DataFrame({"x": [1, 2]}))
    assert ts.isna().all()


def test_cov_attach_ts_empty_dataframe_branch(bmrs):
    out = bmrs.attach_ts(pd.DataFrame())
    assert "ts" in out.columns
    assert out.empty


def test_cov_attach_ts_skips_invalid_first_time_column(bmrs):
    df = pd.DataFrame(
        {
            "startTime": ["bad", "bad"],
            "endTime": ["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"],
            "value": [1, 2],
        }
    )
    out = bmrs.attach_ts(df)
    assert out["ts"].notna().all()


def test_cov_numeric_ts_frame_empty_input(bmrs):
    out = bmrs.numeric_ts_frame(pd.DataFrame(), prefix="x")
    assert out.empty


def test_cov_numeric_ts_frame_w_empty_after_dropna_ts(bmrs):
    df = pd.DataFrame({"startTime": ["invalid", "invalid"], "v": [1, 2]})
    out = bmrs.numeric_ts_frame(df, prefix="x")
    assert out.empty


def test_cov_numeric_ts_frame_no_non_protected_columns(bmrs, monkeypatch):
    def fake_attach(_df):
        return pd.DataFrame({"ts": pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True)})

    monkeypatch.setattr(bmrs, "attach_ts", fake_attach)
    out = bmrs.numeric_ts_frame(pd.DataFrame({"whatever": [1]}), prefix="x")
    assert out.empty


def test_cov_mid_feature_frame_empty_input(bmrs):
    feats, long_df, wide = bmrs.mid_feature_frame(pd.DataFrame(), prefix="price_mid")
    assert feats.empty and long_df.empty and wide.empty


def test_cov_mid_feature_frame_all_ts_nat(bmrs):
    raw = pd.DataFrame({"startTime": ["bad"], "dataProvider": ["APXMIDP"], "price": [100]})
    feats, long_df, wide = bmrs.mid_feature_frame(raw, prefix="price_mid")
    assert feats.empty and long_df.empty and wide.empty


def test_cov_mid_feature_frame_without_price_and_volume_returns_empty_feats_but_long(bmrs):
    raw = pd.DataFrame(
        {
            "startTime": ["2026-01-01T00:00:00Z"],
            "dataProvider": ["APXMIDP"],
            "settlementDate": ["2026-01-01"],
            "settlementPeriod": [1],
        }
    )
    feats, long_df, wide = bmrs.mid_feature_frame(raw, prefix="price_mid")
    assert feats.empty
    assert not long_df.empty
    assert wide.empty


def test_cov_mid_feature_frame_only_price_branch(bmrs):
    raw = pd.DataFrame(
        {
            "startTime": ["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"],
            "dataProvider": ["APXMIDP", "APXMIDP"],
            "price": [100.0, 101.0],
        }
    )
    feats, _, _ = bmrs.mid_feature_frame(raw, prefix="price_mid")
    assert any(c.startswith("price_mid__price") for c in feats.columns)
    assert not any("volume__sum_providers" in c for c in feats.columns)


def test_cov_mid_feature_frame_only_volume_branch(bmrs):
    raw = pd.DataFrame(
        {
            "startTime": ["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"],
            "dataProvider": ["APXMIDP", "APXMIDP"],
            "volume": [10.0, 11.0],
        }
    )
    feats, _, _ = bmrs.mid_feature_frame(raw, prefix="price_mid")
    assert any("volume__sum_providers" in c for c in feats.columns)
    assert not any("price__vwap_providers" in c for c in feats.columns)


def test_cov_mid_feature_frame_disjoint_common_cols_no_vwap(bmrs):
    raw = pd.DataFrame(
        {
            "startTime": ["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
            "dataProvider": ["A", "B"],
            "price": [100.0, np.nan],
            "volume": [np.nan, 10.0],
        }
    )
    feats, _, _ = bmrs.mid_feature_frame(raw, prefix="price_mid")
    assert not any("vwap_providers" in c for c in feats.columns)


def test_cov_resample_15m_empty_input(bmrs):
    out = bmrs.resample_15m(pd.DataFrame(), mode="half_hour")
    assert out.empty


def test_cov_resample_15m_duplicate_index_groupby_branch(bmrs):
    idx = pd.to_datetime(["2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"], utc=True)
    df = pd.DataFrame({"x": [1.0, 3.0]}, index=idx)
    out = bmrs.resample_15m(df, mode="half_hour")
    assert not out.empty
    assert out.index.is_unique


def test_cov_resample_15m_no_numeric_columns_returns_empty(bmrs):
    idx = pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True)
    df = pd.DataFrame({"txt": ["a"]}, index=idx)
    out = bmrs.resample_15m(df, mode="half_hour")
    assert out.empty


def test_cov_choose_target_price_col_contains_price_fallback_line_743(bmrs):
    col = bmrs.choose_target_price_col(["feature_a", "custom_price_signal"])
    assert col == "custom_price_signal"


def test_cov_fetch_endpoint_robust_non_discovery_breaks_on_max_combos(bmrs, monkeypatch):
    cfg = bmrs.PipelineConfig(
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-02T00:00:00Z",
        discovery_chunks=0,
        max_combos_per_chunk=1,
    )
    spec = bmrs.EndpointSpec(
        name="test_ep_break",
        paths=["/a", "/b"],
        mode="half_hour",
        chunk_days=1,
        param_styles=["from_to", "publish_window"],
    )

    calls = []

    def fake_req(_session, _cfg, path, style, t0, t1):
        calls.append((path, style))
        return pd.DataFrame(), {"dummy": 1}

    monkeypatch.setitem(bmrs.REQUEST_FN_BY_ENDPOINT, spec.name, fake_req)

    cache = {spec.name: ("/b", "publish_window")}
    raw, logs = bmrs.fetch_endpoint_robust(None, cfg, spec, cache)

    assert raw.empty
    assert len(calls) == 1
    assert calls[0] == ("/b", "publish_window")
    assert int(logs.iloc[0]["tried"]) == 1


def test_cov_fetch_endpoint_robust_non_discovery_skips_empty_then_success(bmrs, monkeypatch):
    cfg = bmrs.PipelineConfig(
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-02T00:00:00Z",
        discovery_chunks=0,
        max_combos_per_chunk=2,
    )
    spec = bmrs.EndpointSpec(
        name="test_ep_skip_empty",
        paths=["/a", "/b"],
        mode="half_hour",
        chunk_days=1,
        param_styles=["from_to"],
    )

    calls = {"n": 0}

    def fake_req(_session, _cfg, path, style, t0, t1):
        calls["n"] += 1
        if calls["n"] == 1:
            return pd.DataFrame(), {"x": 1}
        return pd.DataFrame({"foo": [1], "startTime": ["2026-01-01T00:00:00Z"]}), {"x": 2}

    monkeypatch.setitem(bmrs.REQUEST_FN_BY_ENDPOINT, spec.name, fake_req)

    cache = {spec.name: ("/a", "from_to")}
    raw, logs = bmrs.fetch_endpoint_robust(None, cfg, spec, cache)

    assert not raw.empty
    assert calls["n"] == 2
    assert int(logs.iloc[0]["rows"]) == 1


def test_cov_fetch_endpoint_robust_drop_duplicates_without_key_candidates(bmrs, monkeypatch):
    cfg = bmrs.PipelineConfig(
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-03T00:00:00Z",  # two chunks with chunk_days=1
        discovery_chunks=0,
        max_combos_per_chunk=1,
    )
    spec = bmrs.EndpointSpec(
        name="test_ep_nokey",
        paths=["/a"],
        mode="half_hour",
        chunk_days=1,
        param_styles=["from_to"],
    )

    def fake_req(_session, _cfg, path, style, t0, t1):
        return pd.DataFrame({"foo": [1]}), {"x": 1}

    monkeypatch.setitem(bmrs.REQUEST_FN_BY_ENDPOINT, spec.name, fake_req)

    raw, _ = bmrs.fetch_endpoint_robust(None, cfg, spec, {})
    assert len(raw) == 2
    assert "foo" in raw.columns


def test_cov_build_pipeline_branches_raw_feature_checkpoint_midwide_and_return_none(
    bmrs, tmp_path, monkeypatch
):
    cfg = bmrs.PipelineConfig(
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-01T01:00:00Z",
        out_dir=str(tmp_path),
        save_raw_per_endpoint=True,
        save_feature_per_endpoint=True,
        save_checkpoint_each_endpoint=True,
        save_final_csv=False,  # cover branch where final CSV is not saved
        return_final_df=False,  # cover branch that returns None
        save_mid_15m_wide=True,
    )

    specs = {
        "price_mid": bmrs.EndpointSpec(
            name="price_mid",
            paths=["/pm"],
            mode="half_hour",
            chunk_days=1,
            param_styles=["from_to"],
        ),
        "demand_itsdo": bmrs.EndpointSpec(
            name="demand_itsdo",
            paths=["/it"],
            mode="half_hour",
            chunk_days=1,
            param_styles=["from_to"],
        ),
    }

    monkeypatch.setattr(bmrs, "build_endpoint_specs", lambda enable_freq=False: specs)

    class DummySession:
        def close(self):
            return None

    monkeypatch.setattr(bmrs, "make_session", lambda: DummySession())

    def fake_fetch(_session, _cfg, spec, _cache):
        if spec.name == "price_mid":
            raw = pd.DataFrame(
                {
                    "startTime": ["2026-01-01T00:00:00Z"],
                    "price": [100.0],
                    "volume": [10.0],
                    "dataProvider": ["APXMIDP"],
                }
            )
        else:
            raw = pd.DataFrame({"startTime": ["2026-01-01T00:00:00Z"], "demand": [1.0]})
        logs = pd.DataFrame([{"endpoint": spec.name, "rows": len(raw)}])
        return raw, logs

    monkeypatch.setattr(bmrs, "fetch_endpoint_robust", fake_fetch)

    # price_mid => non-empty rs (to trigger feature save + checkpoint)
    def fake_build_price_mid(raw, spec):
        idx = pd.to_datetime(["2026-01-01T00:00:00Z"], utc=True)
        rs = pd.DataFrame({"price_mid__price__APXMIDP": [100.0]}, index=idx)
        mid_long = pd.DataFrame(
            {"ts": idx, "dataProvider": ["APXMIDP"], "price": [100.0], "volume": [10.0]}
        )
        mid_wide = pd.DataFrame({"ts": idx, "price__APXMIDP": [100.0], "volume__APXMIDP": [10.0]})
        return rs, mid_long, mid_wide

    # generic => empty rs (to trigger `if rs.empty: continue`)
    def fake_build_generic(raw, spec):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    monkeypatch.setattr(bmrs, "build_features_price_mid", fake_build_price_mid)
    monkeypatch.setattr(bmrs, "build_features_generic", fake_build_generic)

    final, out_paths = bmrs.build_bmrs_dataset_15m_all(cfg)

    assert final is None
    assert "diagnostics" in out_paths

    # raw save branch
    assert (tmp_path / "raw_price_mid.csv").exists()
    # feature save branch for non-empty price_mid
    assert (tmp_path / "feat_price_mid_15m.csv").exists()
    # checkpoint branch
    assert (tmp_path / "dataset_checkpoint.csv").exists()
    # mid wide 15m branch
    assert (tmp_path / "mid_wide_15m.csv").exists()


def test_cov_best_time_series_loop_continue_arc_363_to_360(bmrs):
    """
    Covers branch 363->360:
    first TIME_CANDIDATE column exists but is all invalid dates -> continue loop
    then second candidate is valid and returned.
    """
    df = pd.DataFrame(
        {
            "startTime": ["bad-date", "also-bad"],  # first candidate -> all NaT
            "endTime": ["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"],  # next candidate valid
        }
    )
    ts = bmrs._best_time_series_for_score(df)
    assert ts.notna().all()


def test_cov_fetch_endpoint_discovery_sc_not_improving_arc_444_to_428(bmrs, monkeypatch):
    """
    Covers branch 444->428:
    in discovery mode, first combo sets best_score, second combo has lower score
    so 'if sc > best_score' is False and loop continues.
    """
    cfg = bmrs.PipelineConfig(
        start_utc="2026-01-01T00:00:00Z",
        end_utc="2026-01-02T00:00:00Z",
        discovery_chunks=1,  # ensure discovery=True on first chunk
        max_combos_per_chunk=10,
    )
    spec = bmrs.EndpointSpec(
        name="ep_discovery_branch",
        paths=["/a", "/b"],  # two combos in ordered loop
        mode="half_hour",
        chunk_days=1,
        param_styles=["from_to"],
    )

    calls = []

    def fake_request_fn(_session, _cfg, path, style, t0, t1):
        calls.append((path, style))
        if path == "/a":
            # Higher score
            df = pd.DataFrame(
                {
                    "startTime": ["2026-01-01T00:00:00Z", "2026-01-01T00:30:00Z"],
                    "x": [1, 2],
                }
            )
        else:
            # Lower score -> should NOT replace best
            df = pd.DataFrame({"startTime": ["2026-01-01T00:00:00Z"], "x": [1]})
        return df, {"dummy": True}

    monkeypatch.setitem(bmrs.REQUEST_FN_BY_ENDPOINT, spec.name, fake_request_fn)

    raw, logs = bmrs.fetch_endpoint_robust(
        session=None,
        cfg=cfg,
        spec=spec,
        best_strategy_cache={},  # no cache -> discovery path
    )

    assert len(calls) == 2
    assert not raw.empty
    # best should come from first combo /a
    assert (raw["__path"] == "/a").any()
    assert int(logs.iloc[0]["tried"]) == 2


def test_cov_choose_target_price_col_hits_line_743_fallback_return(bmrs):
    """
    Covers line 743 (fallback loop return):
    force first 'price' check in priorities loop to be False,
    and second 'price' check in fallback loop to be True.
    """

    class ToggleContains:
        def __init__(self):
            self.price_seen = 0

        def __contains__(self, item):
            # Return False first time we see 'price' (inside priorities loop),
            # and True second time (fallback loop at line 743).
            if item == "price":
                self.price_seen += 1
                return self.price_seen >= 2
            return False

    class WeirdCol(str):
        def __new__(cls, value, container):
            obj = str.__new__(cls, value)
            obj._container = container
            return obj

        def lower(self):
            # Return custom container instead of a plain str
            return self._container

    token = ToggleContains()
    col = WeirdCol("any_name", token)

    chosen = bmrs.choose_target_price_col([col])

    assert chosen == col
    assert token.price_seen >= 2


# =========================
# NEW TESTS: price_data extraction/export
# =========================


def _get_required_fn(module, fn_name: str):
    assert hasattr(module, fn_name), (
        f"El módulo no tiene la función '{fn_name}'. "
        f"Asegúrate de haber actualizado src/ors/services/price_api.py"
    )
    return getattr(module, fn_name)


def test_extract_price_model_frame_happy_path(bmrs):
    fn = _get_required_fn(bmrs, "extract_price_model_frame")

    df = pd.DataFrame(
        {
            "ts_utc": ["2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z"],
            "price_mid__price__APXMIDP": [100.5, 101.0],
            "demand_itsdo__demand": [30000, 30100],
            "demand_indo__initialDemandOutturn": [29900, 30050],
            "demand_inddem__demand": [29800, 29950],
            "other_col": [1, 2],
        }
    )

    out = fn(df)

    assert "price" in out.columns
    assert "demand_itsdo" in out.columns
    assert "timestamp" in out.columns
    assert "demand_indo" in out.columns

    assert any(c in out.columns for c in ["demand_inddem", "demand_indeem"])

    assert len(out) == 2
    assert float(out["price"].iloc[0]) == 100.5
    assert out["timestamp"].iloc[0] == "2026-01-01T00:00:00Z"


def test_extract_price_model_frame_strict_true_raises_when_missing_cols(bmrs):
    fn = _get_required_fn(bmrs, "extract_price_model_frame")
    sig = inspect.signature(fn)

    if "strict" not in sig.parameters:
        pytest.skip("extract_price_model_frame no tiene parámetro 'strict'.")

    # Falta demand_indo__initialDemandOutturn y demand_inddem__demand
    df_missing = pd.DataFrame(
        {
            "ts_utc": ["2026-01-01T00:00:00Z"],
            "price_mid__price__APXMIDP": [100.0],
            "demand_itsdo__demand": [30000],
        }
    )

    with pytest.raises((KeyError, ValueError)):
        fn(df_missing, strict=True)


def test_extract_price_model_frame_strict_false_tolerates_missing_cols(bmrs):
    fn = _get_required_fn(bmrs, "extract_price_model_frame")
    sig = inspect.signature(fn)

    if "strict" not in sig.parameters:
        pytest.skip("extract_price_model_frame no tiene parámetro 'strict'.")

    df_missing = pd.DataFrame(
        {
            "ts_utc": ["2026-01-01T00:00:00Z"],
            "price_mid__price__APXMIDP": [100.0],
            "demand_itsdo__demand": [30000],
        }
    )

    out = fn(df_missing, strict=False)
    assert len(out) == 1
    assert "price" in out.columns
    assert "timestamp" in out.columns

    # Si la columna existe, debería venir NaN cuando faltaba origen
    for candidate in ("demand_indo", "demand_inddem", "demand_indeem"):
        if candidate in out.columns:
            assert out[candidate].isna().all()


def test_export_price_model_csv_from_merged_creates_file(tmp_path, bmrs):
    fn = _get_required_fn(bmrs, "export_price_model_csv_from_merged")

    in_csv = tmp_path / "merged.csv"
    out_csv = tmp_path / "price_data.csv"

    src = pd.DataFrame(
        {
            "ts_utc": ["2026-01-01T00:00:00Z", "2026-01-01T00:15:00Z"],
            "price_mid__price__APXMIDP": [100.0, 101.0],
            "demand_itsdo__demand": [30000, 30100],
            "demand_indo__initialDemandOutturn": [29900, 30050],
            "demand_inddem__demand": [29800, 29950],
        }
    )
    src.to_csv(in_csv, index=False)

    ret = fn(str(in_csv), str(out_csv))
    assert out_csv.exists()

    out = pd.read_csv(out_csv)
    assert len(out) == 2
    assert "price" in out.columns
    assert "timestamp" in out.columns
    assert "demand_itsdo" in out.columns
    assert "demand_indo" in out.columns
    assert any(c in out.columns for c in ["demand_inddem", "demand_indeem"])

    # retorno flexible (por si devuelves path o dataframe o None)
    assert (ret is None) or isinstance(ret, (str, pd.DataFrame))


def test_export_price_model_csv_from_merged_missing_input_raises(tmp_path, bmrs):
    fn = _get_required_fn(bmrs, "export_price_model_csv_from_merged")

    missing_input = tmp_path / "does_not_exist.csv"
    out_csv = tmp_path / "price_data.csv"

    with pytest.raises(FileNotFoundError):
        fn(str(missing_input), str(out_csv))


def test_main_contains_call_to_export_price_model_csv_from_merged(bmrs):
    """
    Test estático sencillo para comprobar que el flujo principal incluye
    la llamada a la exportación del CSV reducido.
    """
    src = inspect.getsource(bmrs)
    assert "export_price_model_csv_from_merged" in src
    assert "price_data.csv" in src
