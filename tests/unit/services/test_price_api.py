from datetime import datetime, timezone

import pandas as pd
import pytest
import requests
import src.ors.services.price_api.price_api as pa
import src.ors.services.price_api.price_api_hist as pah


class DummyResponse:
    def __init__(self, status_code=200, json_data=None, text="OK"):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text

    def json(self):
        return self._json_data


class DummySession:
    def __init__(self, response: DummyResponse):
        self._response = response
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params, "timeout": timeout})
        return self._response

    def close(self):
        self.closed = True


def test_make_session_has_retry_adapter():
    s = pa.make_session()
    try:
        assert "https://" in s.adapters
        assert "http://" in s.adapters
        https_adapter = s.adapters["https://"]
        assert hasattr(https_adapter, "max_retries")
        mr = https_adapter.max_retries
        assert mr.total == 3
        assert 429 in mr.status_forcelist
        assert "GET" in mr.allowed_methods
    finally:
        s.close()


@pytest.mark.parametrize(
    "payload, expected_cols",
    [
        ([{"a": 1}, {"a": 2}], ["a"]),
        ({"data": [{"a": 1}]}, ["a"]),
        ({"results": [{"a": 1}]}, ["a"]),
        ({"result": [{"a": 1}]}, ["a"]),
        ({"items": [{"a": 1}]}, ["a"]),
        ({"records": [{"a": 1}]}, ["a"]),
        ({"x": [{"a": 1}]}, ["a"]),
        ({"a": 1, "b": 2}, ["a", "b"]),
        ("not-json", []),
    ],
)
def test_normalize_payload_branches(payload, expected_cols):
    df = pa._normalize_payload(payload)
    if expected_cols:
        for c in expected_cols:
            assert c in df.columns
    else:
        assert df.empty


def test_bmrs_get_success_normalizes():
    sess = DummySession(DummyResponse(200, json_data=[{"x": 1}, {"x": 2}]))
    df = pa.bmrs_get(sess, "https://base", "/p", {"a": 1}, timeout=12)
    assert list(df["x"]) == [1, 2]
    assert sess.calls[0]["url"] == "https://base/p"
    assert sess.calls[0]["params"] == {"a": 1}
    assert sess.calls[0]["timeout"] == 12


def test_bmrs_get_http_error():
    sess = DummySession(DummyResponse(500, json_data=None, text="ERRTEXT"))
    with pytest.raises(requests.HTTPError) as ei:
        pa.bmrs_get(sess, "https://base", "/p", {"a": 1})
    msg = str(ei.value)
    assert "500" in msg
    assert "https://base/p" in msg
    assert "ERRTEXT" in msg


def test_time_helpers_to_dt_utc_string_and_datetime():
    dt = pa.to_dt_utc("2025-01-01T12:34:56+02:00")
    assert dt.tzinfo is not None
    assert dt.astimezone(timezone.utc).hour == 10

    dt2 = pa.to_dt_utc(datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc))
    assert dt2.tzinfo is not None
    assert dt2.isoformat().endswith("+00:00")


def test_time_helpers_iso_formats():
    dt = datetime(2025, 1, 1, 0, 0, 59, tzinfo=timezone.utc)
    assert pa.to_isoz(dt).endswith("Z")
    assert pa.to_rfc3339_minute_z(dt) == "2025-01-01T00:00Z"


def test_param_helpers():
    start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 1, 1, 1, 0, tzinfo=timezone.utc)

    p1 = pa._params_from_to(start, end)
    assert p1["from"].endswith("Z") and p1["to"].endswith("Z")

    p2 = pa._params_publish_window(start, end)
    assert p2["publishDateTimeFrom"].endswith("Z") and p2["publishDateTimeTo"].endswith("Z")

    p3 = pa._params_settlement_window(start, end)
    assert p3["settlementDateFrom"] == "2025-01-01"
    assert p3["settlementDateTo"] == "2025-01-01"


def test_bmrs_get_first_success_returns_first_non_empty(monkeypatch):
    calls = []

    def fake_bmrs_get(session, base_url, path, params, timeout=60):
        calls.append(params)
        if params.get("k") == 1:
            return pd.DataFrame()
        return pd.DataFrame([{"a": 1}])

    monkeypatch.setattr(pa, "bmrs_get", fake_bmrs_get)

    out = pa.bmrs_get_first_success(
        session=object(),
        base_url="b",
        path="/x",
        params_candidates=[{"k": 1}, {"k": 2}],
    )
    assert not out.empty
    assert out.iloc[0]["a"] == 1
    assert calls == [{"k": 1}, {"k": 2}]


def test_bmrs_get_first_success_returns_first_empty_when_all_empty(monkeypatch):
    monkeypatch.setattr(pa, "bmrs_get", lambda *a, **k: pd.DataFrame())

    out = pa.bmrs_get_first_success(
        session=object(),
        base_url="b",
        path="/x",
        params_candidates=[{"k": 1}, {"k": 2}],
    )
    assert out.empty


def test_bmrs_get_first_success_raises_last_error_when_all_error(monkeypatch):
    class BoomError(Exception):
        pass

    def fake_bmrs_get(session, base_url, path, params, timeout=60):
        raise BoomError(f"boom_error {params}")

    monkeypatch.setattr(pa, "bmrs_get", fake_bmrs_get)

    with pytest.raises(BoomError):
        pa.bmrs_get_first_success(
            session=object(),
            base_url="b",
            path="/x",
            params_candidates=[{"k": 1}, {"k": 2}],
        )


def test_extract_ts_utc_settlement_preferred():
    raw = pd.DataFrame({"settlementDate": ["2025-01-01", "2025-01-01"], "settlementPeriod": [1, 2]})
    ts = pa._extract_ts_utc(raw)

    assert ts.notna().all()
    assert ts.iloc[0] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert ts.iloc[1] == pd.Timestamp("2025-01-01T00:30:00Z")


def test_extract_ts_utc_fallback_columns_and_all_nat():
    raw = pd.DataFrame({"startTime": ["2025-01-01T00:00:00Z", None]})
    ts = pa._extract_ts_utc(raw)
    assert ts.notna().any()

    raw2 = pd.DataFrame({"startTime": [None, None]})
    ts2 = pa._extract_ts_utc(raw2)
    assert ts2.isna().all()


def test_extract_ts_from_settlement_missing_cols_returns_nat():
    raw = pd.DataFrame({"x": [1]})
    ts = pa._extract_ts_from_settlement(raw)
    assert ts.isna().all()


def test_first_existing_column_found_and_not_found():
    raw = pd.DataFrame({"b": [1, 2]})
    s = pa._first_existing_column(raw, ["a", "b", "c"])
    assert list(s) == [1, 2]

    s2 = pa._first_existing_column(raw, ["a", "c"])
    assert s2.isna().all()


def test_fetch_mid_price_empty_raw(monkeypatch):
    monkeypatch.setattr(pa, "bmrs_get", lambda *a, **k: pd.DataFrame())

    out = pa.fetch_mid_price(
        session=object(),
        base_url="b",
        start_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2025, 1, 2, tzinfo=timezone.utc),
        data_providers=[],
    )
    assert list(out.columns) == ["ts_utc", "price"]
    assert out.empty


def test_fetch_mid_price_ts_all_na(monkeypatch):
    monkeypatch.setattr(pa, "bmrs_get", lambda *a, **k: pd.DataFrame([{"price": 10}]))
    monkeypatch.setattr(pa, "_extract_ts_utc", lambda raw: pd.Series([pd.NaT]))

    out = pa.fetch_mid_price(
        session=object(),
        base_url="b",
        start_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2025, 1, 2, tzinfo=timezone.utc),
        data_providers=[],
    )
    assert out.empty


def test_fetch_mid_price_happy_path_dedup_and_provider_param(monkeypatch):
    captured = {}

    def fake_bmrs_get(session, base_url, path, params, timeout=60):
        captured["params"] = params
        return pd.DataFrame(
            {
                "settlementDate": ["2025-01-01", "2025-01-01"],
                "settlementPeriod": [1, 1],
                "price": ["10.5", "11.0"],
            }
        )

    monkeypatch.setattr(pa, "bmrs_get", fake_bmrs_get)

    out = pa.fetch_mid_price(
        session=object(),
        base_url="b",
        start_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2025, 1, 2, tzinfo=timezone.utc),
        data_providers=["APXMIDP"],
    )
    assert captured["params"]["dataProviders"] == ["APXMIDP"]
    assert len(out) == 1
    assert float(out.iloc[0]["price"]) == pytest.approx(11.0)


def test_fetch_itsdo_demand_empty_raw(monkeypatch):
    monkeypatch.setattr(pa, "bmrs_get_first_success", lambda **k: pd.DataFrame())

    out = pa.fetch_itsdo_demand(
        session=object(),
        base_url="b",
        start_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    assert out.empty
    assert list(out.columns) == ["ts_utc", "demand_itsdo"]


def test_fetch_itsdo_demand_ts_fallback_then_settlement_reapplied(monkeypatch):
    # Force the first settlement extraction to return NaT so the function falls back to _extract_ts_utc.
    raw = pd.DataFrame(
        {"settlementDate": ["2025-01-01"], "settlementPeriod": [1], "demand": ["100"]}
    )
    monkeypatch.setattr(pa, "bmrs_get_first_success", lambda **k: raw)

    calls = {"sett": 0}
    orig_settlement = pa._extract_ts_from_settlement

    def settlement_proxy(df):
        calls["sett"] += 1
        if calls["sett"] == 1:
            return pd.Series([pd.NaT], index=df.index, dtype="datetime64[ns, UTC]")
        return orig_settlement(df)

    monkeypatch.setattr(pa, "_extract_ts_from_settlement", settlement_proxy)
    monkeypatch.setattr(
        pa, "_extract_ts_utc", lambda df: pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True)
    )

    out = pa.fetch_itsdo_demand(
        session=object(),
        base_url="b",
        start_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    assert len(out) == 1
    assert float(out.iloc[0]["demand_itsdo"]) == pytest.approx(100.0)


def test_fetch_itsdo_demand_all_ts_na_returns_empty(monkeypatch):
    raw = pd.DataFrame({"demand": [1]})
    monkeypatch.setattr(pa, "bmrs_get_first_success", lambda **k: raw)
    monkeypatch.setattr(
        pa,
        "_extract_ts_from_settlement",
        lambda df: pd.Series([pd.NaT], index=df.index, dtype="datetime64[ns, UTC]"),
    )
    monkeypatch.setattr(
        pa,
        "_extract_ts_utc",
        lambda df: pd.Series([pd.NaT], index=df.index, dtype="datetime64[ns, UTC]"),
    )

    out = pa.fetch_itsdo_demand(
        session=object(),
        base_url="b",
        start_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    assert out.empty


@pytest.mark.parametrize(
    "fn,out_col",
    [
        (pa.fetch_indo_initial_demand, "demand_indo"),
        (pa.fetch_inddem_demand, "demand_inddem"),
    ],
)
def test_fetch_indo_and_inddem_choose_first_existing_column(monkeypatch, fn, out_col):
    raw = pd.DataFrame({"settlementDate": ["2025-01-01"], "settlementPeriod": [1], "value": ["7"]})
    monkeypatch.setattr(pa, "bmrs_get_first_success", lambda **k: raw)

    out = fn(
        session=object(),
        base_url="b",
        start_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    assert len(out) == 1
    assert out_col in out.columns
    assert float(out.iloc[0][out_col]) == pytest.approx(7.0)


def test_fetch_current_price_data_merged_empty(monkeypatch):
    monkeypatch.setattr(
        pa, "fetch_mid_price", lambda **k: pd.DataFrame(columns=["ts_utc", "price"])
    )
    monkeypatch.setattr(
        pa, "fetch_itsdo_demand", lambda **k: pd.DataFrame(columns=["ts_utc", "demand_itsdo"])
    )
    monkeypatch.setattr(
        pa,
        "fetch_indo_initial_demand",
        lambda **k: pd.DataFrame(columns=["ts_utc", "demand_indo"]),
    )
    monkeypatch.setattr(
        pa,
        "fetch_inddem_demand",
        lambda **k: pd.DataFrame(columns=["ts_utc", "demand_inddem"]),
    )

    out = pa.fetch_current_price_data(session=object(), base_url="b", data_providers=[])
    assert list(out.columns) == [
        "price",
        "demand_itsdo",
        "timestamp",
        "demand_indo",
        "demand_inddem",
    ]
    assert out.empty


def test_fetch_current_price_data_merges_and_sorts(monkeypatch):
    t0 = pd.Timestamp("2025-01-01T00:00:00Z")
    t1 = pd.Timestamp("2025-01-01T00:30:00Z")
    mid = pd.DataFrame({"ts_utc": [t1, t0], "price": [2, 1]})
    itsdo = pd.DataFrame({"ts_utc": [t0], "demand_itsdo": [10]})
    indo = pd.DataFrame({"ts_utc": [t0], "demand_indo": [20]})
    inddem = pd.DataFrame({"ts_utc": [t1], "demand_inddem": [30]})

    monkeypatch.setattr(pa, "fetch_mid_price", lambda **k: mid)
    monkeypatch.setattr(pa, "fetch_itsdo_demand", lambda **k: itsdo)
    monkeypatch.setattr(pa, "fetch_indo_initial_demand", lambda **k: indo)
    monkeypatch.setattr(pa, "fetch_inddem_demand", lambda **k: inddem)

    out = pa.fetch_current_price_data(session=object(), base_url="b", data_providers=[])
    assert list(out["timestamp"]) == [t0, t1]
    assert out.loc[0, "price"] == 1
    assert out.loc[0, "demand_itsdo"] == 10
    assert out.loc[0, "demand_indo"] == 20
    assert pd.isna(out.loc[0, "demand_inddem"])
    assert out.loc[1, "price"] == 2
    assert out.loc[1, "demand_inddem"] == 30


def test_history_builder_init_sets_mid_providers_and_master_index(tmp_path):
    cfg = pah.HistoryConfig(
        base_url="b",
        start_utc="2025-01-01T00:00:00Z",
        end_utc="2025-01-01T01:00:00Z",
        out_dir=str(tmp_path),
        mid_providers=None,
        freq="15min",
    )
    b = pah.PriceHistoryBuilder(cfg)
    assert b.cfg.mid_providers == []
    assert len(b.master_index) == 4
    assert str(b.master_index.tz) == "UTC"


def test_safe_call_continue_on_error_true_returns_empty_and_warns(tmp_path, capsys):
    cfg = pah.HistoryConfig(
        base_url="b",
        start_utc="2025-01-01T00:00:00Z",
        end_utc="2025-01-01T00:15:00Z",
        out_dir=str(tmp_path),
        continue_on_error=True,
    )
    b = pah.PriceHistoryBuilder(cfg)

    def boom_error():
        raise RuntimeError("nope")

    out = b._safe_call(boom_error)
    assert out.empty
    assert "[WARN] Call failed" in capsys.readouterr().out


def test_safe_call_continue_on_error_false_raises(tmp_path):
    cfg = pah.HistoryConfig(
        base_url="b",
        start_utc="2025-01-01T00:00:00Z",
        end_utc="2025-01-01T00:15:00Z",
        out_dir=str(tmp_path),
        continue_on_error=False,
    )
    b = pah.PriceHistoryBuilder(cfg)

    def boom_error():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        b._safe_call(boom_error)


def test_fetch_in_chunks_appends_only_non_empty_and_logs_every_10(tmp_path, capsys):
    cfg = pah.HistoryConfig(
        base_url="b",
        start_utc="2025-01-01T00:00:00Z",
        end_utc="2025-01-12T00:00:00Z",
        out_dir=str(tmp_path),
        continue_on_error=True,
    )
    b = pah.PriceHistoryBuilder(cfg)

    def safe_call(fetch_fn, *args, **kwargs):
        t0 = args[2]
        if t0.day % 2 == 0:
            return pd.DataFrame()
        return pd.DataFrame({"ts_utc": [pd.Timestamp(t0)], "price": [1]})

    b._safe_call = safe_call

    out = b._fetch_in_chunks(
        session=object(),
        chunk_days=1,
        fetch_fn=lambda *a, **k: pd.DataFrame(),
        fetch_args_builder=lambda t0, t1: ((object(), "b", t0, t1, [], 60), {}),
        label="X",
    )

    assert ("price" in out.columns) or (not out.empty)
    assert "[X]" in capsys.readouterr().out


def test_to_series_frame_branches_and_dedup(tmp_path):
    cfg = pah.HistoryConfig(
        base_url="b",
        start_utc="2025-01-01T00:00:00Z",
        end_utc="2025-01-01T01:00:00Z",
        out_dir=str(tmp_path),
    )
    b = pah.PriceHistoryBuilder(cfg)

    assert b._to_series_frame(pd.DataFrame(), "price", "price").empty
    assert b._to_series_frame(pd.DataFrame({"x": [1]}), "price", "price").empty

    df = pd.DataFrame(
        {
            "ts_utc": ["bad", "2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"],
            "price": ["1", "2", "3"],
        }
    )
    out = b._to_series_frame(df, "price", "price")
    assert list(out.columns) == ["price"]
    assert len(out) == 1
    assert float(out.iloc[0]["price"]) == pytest.approx(3.0)


def test_resample_half_hour_to_15m(tmp_path):
    cfg = pah.HistoryConfig(
        base_url="b",
        start_utc="2025-01-01T00:00:00Z",
        end_utc="2025-01-01T01:00:00Z",
        out_dir=str(tmp_path),
    )
    b = pah.PriceHistoryBuilder(cfg)

    idx = pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:30:00Z"], utc=True)
    frame = pd.DataFrame({"price": [1.0, 2.0]}, index=idx)

    out = b._resample_half_hour_to_15m(frame)
    assert out.loc[pd.Timestamp("2025-01-01T00:15:00Z"), "price"] == pytest.approx(1.0)
    assert pd.isna(out.loc[pd.Timestamp("2025-01-01T00:45:00Z"), "price"])


def test_build_end_to_end_saves_csv(tmp_path, monkeypatch):
    cfg = pah.HistoryConfig(
        base_url="b",
        start_utc="2025-01-01T00:00:00Z",
        end_utc="2025-01-01T01:00:00Z",
        out_dir=str(tmp_path),
        output_csv="price_data.csv",
        continue_on_error=True,
    )
    b = pah.PriceHistoryBuilder(cfg)

    sess = DummySession(DummyResponse())
    monkeypatch.setattr(pah, "make_session", lambda: sess)

    t0 = pd.Timestamp("2025-01-01T00:00:00Z")
    t1 = pd.Timestamp("2025-01-01T00:30:00Z")
    mid_raw = pd.DataFrame({"ts_utc": [t0, t1], "price": [1, 2]})
    itsdo_raw = pd.DataFrame({"ts_utc": [t0, t1], "demand_itsdo": [10, 20]})
    indo_raw = pd.DataFrame({"ts_utc": [t0], "demand_indo": [30]})
    inddem_raw = pd.DataFrame({"ts_utc": [t1], "demand_inddem": [40]})

    def fake_fetch_in_chunks(session, chunk_days, fetch_fn, fetch_args_builder, label):
        return {"MID": mid_raw, "ITSDO": itsdo_raw, "INDO": indo_raw, "INDDEM": inddem_raw}[label]

    monkeypatch.setattr(b, "_fetch_in_chunks", fake_fetch_in_chunks)

    out = b.build()
    assert list(out.columns) == [
        "timestamp",
        "price",
        "demand_itsdo",
        "demand_indo",
        "demand_inddem",
    ]
    assert len(out) == 4

    assert (tmp_path / "price_data.csv").exists()
    assert getattr(sess, "closed", False) is True


def test_fetch_itsdo_demand_settlement_all_nat_then_fallback_to_extract_ts_utc(monkeypatch):
    raw = pd.DataFrame({"demand": ["100"]})
    monkeypatch.setattr(pa, "bmrs_get_first_success", lambda **k: raw)

    calls = {"sett": 0, "utc": 0}

    def settlement(df):
        calls["sett"] += 1
        if calls["sett"] == 1:
            return pd.Series([pd.NaT], index=df.index, dtype="datetime64[ns, UTC]")
        return pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True)

    def utc(df):
        calls["utc"] += 1
        return pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True)

    monkeypatch.setattr(pa, "_extract_ts_from_settlement", settlement)
    monkeypatch.setattr(pa, "_extract_ts_utc", utc)

    out = pa.fetch_itsdo_demand(
        session=object(),
        base_url="b",
        start_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    assert len(out) == 1
    assert calls["utc"] == 1


def test_fetch_indo_initial_demand_raw_empty_returns_empty_cols(monkeypatch):
    monkeypatch.setattr(pa, "bmrs_get_first_success", lambda **k: pd.DataFrame())

    out = pa.fetch_indo_initial_demand(
        session=object(),
        base_url="b",
        start_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    assert out.empty
    assert list(out.columns) == ["ts_utc", "demand_indo"]


def test_fetch_indo_initial_demand_settlement_nat_uses_extract_ts_utc(monkeypatch):
    raw = pd.DataFrame({"value": ["7"]})
    monkeypatch.setattr(pa, "bmrs_get_first_success", lambda **k: raw)
    monkeypatch.setattr(
        pa,
        "_extract_ts_from_settlement",
        lambda df: pd.Series([pd.NaT], index=df.index, dtype="datetime64[ns, UTC]"),
    )
    monkeypatch.setattr(
        pa, "_extract_ts_utc", lambda df: pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True)
    )

    out = pa.fetch_indo_initial_demand(
        session=object(),
        base_url="b",
        start_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    assert len(out) == 1
    assert float(out.iloc[0]["demand_indo"]) == pytest.approx(7.0)


def test_fetch_indo_initial_demand_all_ts_nat_returns_empty(monkeypatch):
    raw = pd.DataFrame({"value": ["7"]})
    monkeypatch.setattr(pa, "bmrs_get_first_success", lambda **k: raw)
    monkeypatch.setattr(
        pa,
        "_extract_ts_from_settlement",
        lambda df: pd.Series([pd.NaT], index=df.index, dtype="datetime64[ns, UTC]"),
    )
    monkeypatch.setattr(
        pa,
        "_extract_ts_utc",
        lambda df: pd.Series([pd.NaT], index=df.index, dtype="datetime64[ns, UTC]"),
    )

    out = pa.fetch_indo_initial_demand(
        session=object(),
        base_url="b",
        start_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    assert out.empty
    assert list(out.columns) == ["ts_utc", "demand_indo"]


def test_fetch_inddem_demand_raw_empty_returns_empty_cols(monkeypatch):
    monkeypatch.setattr(pa, "bmrs_get_first_success", lambda **k: pd.DataFrame())

    out = pa.fetch_inddem_demand(
        session=object(),
        base_url="b",
        start_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    assert out.empty
    assert list(out.columns) == ["ts_utc", "demand_inddem"]


def test_fetch_inddem_demand_settlement_nat_uses_extract_ts_utc(monkeypatch):
    raw = pd.DataFrame({"value": ["9"]})
    monkeypatch.setattr(pa, "bmrs_get_first_success", lambda **k: raw)
    monkeypatch.setattr(
        pa,
        "_extract_ts_from_settlement",
        lambda df: pd.Series([pd.NaT], index=df.index, dtype="datetime64[ns, UTC]"),
    )
    monkeypatch.setattr(
        pa, "_extract_ts_utc", lambda df: pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True)
    )

    out = pa.fetch_inddem_demand(
        session=object(),
        base_url="b",
        start_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    assert len(out) == 1
    assert float(out.iloc[0]["demand_inddem"]) == pytest.approx(9.0)


def test_fetch_inddem_demand_all_ts_nat_returns_empty(monkeypatch):
    raw = pd.DataFrame({"value": ["9"]})
    monkeypatch.setattr(pa, "bmrs_get_first_success", lambda **k: raw)
    monkeypatch.setattr(
        pa,
        "_extract_ts_from_settlement",
        lambda df: pd.Series([pd.NaT], index=df.index, dtype="datetime64[ns, UTC]"),
    )
    monkeypatch.setattr(
        pa,
        "_extract_ts_utc",
        lambda df: pd.Series([pd.NaT], index=df.index, dtype="datetime64[ns, UTC]"),
    )

    out = pa.fetch_inddem_demand(
        session=object(),
        base_url="b",
        start_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    assert out.empty
    assert list(out.columns) == ["ts_utc", "demand_inddem"]


def test_fetch_current_price_data_merge_keeps_a_when_b_empty(monkeypatch):
    t0 = pd.Timestamp("2025-01-01T00:00:00Z")
    mid = pd.DataFrame({"ts_utc": [t0], "price": [1.0]})
    empty_itsdo = pd.DataFrame(columns=["ts_utc", "demand_itsdo"])
    empty_indo = pd.DataFrame(columns=["ts_utc", "demand_indo"])
    empty_inddem = pd.DataFrame(columns=["ts_utc", "demand_inddem"])

    monkeypatch.setattr(pa, "fetch_mid_price", lambda **k: mid)
    monkeypatch.setattr(pa, "fetch_itsdo_demand", lambda **k: empty_itsdo)
    monkeypatch.setattr(pa, "fetch_indo_initial_demand", lambda **k: empty_indo)
    monkeypatch.setattr(pa, "fetch_inddem_demand", lambda **k: empty_inddem)

    out = pa.fetch_current_price_data(session=object(), base_url="b", data_providers=[])
    assert len(out) == 1
    assert out.loc[0, "price"] == 1.0


def test_history_builder_init_keeps_mid_providers_when_given(tmp_path):
    cfg = pah.HistoryConfig(
        base_url="b",
        start_utc="2025-01-01T00:00:00Z",
        end_utc="2025-01-01T01:00:00Z",
        out_dir=str(tmp_path),
        mid_providers=["APXMIDP"],
    )
    b = pah.PriceHistoryBuilder(cfg)
    assert b.cfg.mid_providers == ["APXMIDP"]


def test_fetch_in_chunks_all_empty_returns_empty(tmp_path):
    cfg = pah.HistoryConfig(
        base_url="b",
        start_utc="2025-01-01T00:00:00Z",
        end_utc="2025-01-03T00:00:00Z",
        out_dir=str(tmp_path),
        continue_on_error=True,
    )
    b = pah.PriceHistoryBuilder(cfg)

    b._safe_call = lambda fetch_fn, *a, **k: pd.DataFrame()

    out = b._fetch_in_chunks(
        session=object(),
        chunk_days=1,
        fetch_fn=lambda *a, **k: pd.DataFrame(),
        fetch_args_builder=lambda t0, t1: ((object(), "b", t0, t1, [], 60), {}),
        label="MID",
    )
    assert out.empty


def test_to_series_frame_valid_path_executes_numeric_and_dedup(tmp_path):
    cfg = pah.HistoryConfig(
        base_url="b",
        start_utc="2025-01-01T00:00:00Z",
        end_utc="2025-01-01T01:00:00Z",
        out_dir=str(tmp_path),
    )
    b = pah.PriceHistoryBuilder(cfg)

    df = pd.DataFrame(
        {"ts_utc": ["2025-01-01T00:00:00Z", "2025-01-01T00:00:00Z"], "price": ["1", "2"]}
    )
    out = b._to_series_frame(df, "price", "price")
    assert len(out) == 1
    assert float(out.iloc[0]["price"]) == pytest.approx(2.0)


def test_resample_half_hour_to_15m_non_empty_path(tmp_path):
    cfg = pah.HistoryConfig(
        base_url="b",
        start_utc="2025-01-01T00:00:00Z",
        end_utc="2025-01-01T01:00:00Z",
        out_dir=str(tmp_path),
    )
    b = pah.PriceHistoryBuilder(cfg)

    idx = pd.to_datetime(["2025-01-01T00:00:00Z", "2025-01-01T00:30:00Z"], utc=True)
    frame = pd.DataFrame({"price": [1.0, 2.0]}, index=idx)

    out = b._resample_half_hour_to_15m(frame)
    assert out.loc[pd.Timestamp("2025-01-01T00:15:00Z"), "price"] == pytest.approx(1.0)
    assert pd.isna(out.loc[pd.Timestamp("2025-01-01T00:45:00Z"), "price"])


def test_build_adds_missing_columns_and_reorders(tmp_path, monkeypatch):
    cfg = pah.HistoryConfig(
        base_url="b",
        start_utc="2025-01-01T00:00:00Z",
        end_utc="2025-01-01T01:00:00Z",
        out_dir=str(tmp_path),
        continue_on_error=True,
        output_csv="out.csv",
    )
    b = pah.PriceHistoryBuilder(cfg)

    class DummySess:
        def close(self):
            pass

    monkeypatch.setattr(pah, "make_session", lambda: DummySess())

    t0 = pd.Timestamp("2025-01-01T00:00:00Z")
    t1 = pd.Timestamp("2025-01-01T00:30:00Z")
    mid_raw = pd.DataFrame({"ts_utc": [t0, t1], "price": [1, 2]})

    def fake_fetch_in_chunks(session, chunk_days, fetch_fn, fetch_args_builder, label):
        if label == "MID":
            return mid_raw
        return pd.DataFrame()

    monkeypatch.setattr(b, "_fetch_in_chunks", fake_fetch_in_chunks)

    out = b.build()
    assert list(out.columns) == [
        "timestamp",
        "price",
        "demand_itsdo",
        "demand_indo",
        "demand_inddem",
    ]
    assert (tmp_path / "out.csv").exists()


def test_fetch_itsdo_demand_settlement_ok_skips_extract_ts_utc(monkeypatch):
    raw = pd.DataFrame(
        {"settlementDate": ["2025-01-01"], "settlementPeriod": [1], "demand": ["123"]}
    )
    monkeypatch.setattr(pa, "bmrs_get_first_success", lambda **k: raw)

    # If this gets called, the branch was not taken as expected.
    monkeypatch.setattr(
        pa,
        "_extract_ts_utc",
        lambda df: (_ for _ in ()).throw(AssertionError("_extract_ts_utc should not be called")),
    )

    out = pa.fetch_itsdo_demand(
        session=object(),
        base_url="b",
        start_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        end_utc=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )

    assert len(out) == 1
    assert float(out.iloc[0]["demand_itsdo"]) == pytest.approx(123.0)


def test_to_series_frame_returns_empty_when_all_ts_invalid(tmp_path):
    cfg = pah.HistoryConfig(
        base_url="b",
        start_utc="2025-01-01T00:00:00Z",
        end_utc="2025-01-01T01:00:00Z",
        out_dir=str(tmp_path),
    )
    b = pah.PriceHistoryBuilder(cfg)

    df = pd.DataFrame({"ts_utc": ["BAD_TS_1", "BAD_TS_2"], "price": ["1.0", "2.0"]})

    out = b._to_series_frame(df, value_col="price", out_col="price")
    assert out.empty
