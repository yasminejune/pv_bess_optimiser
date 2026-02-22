from __future__ import annotations

import pandas as pd
import pytest

from ors.domain.models.pv import PVSpec
from ors.services import weather_to_pv as integ


def _mk_spec() -> PVSpec:
    # Needs panel_area_m2 + panel_efficiency so PV can estimate from radiance
    return PVSpec(
        rated_power_kw=9999.0,
        max_export_kw=None,
        panel_area_m2=10.0,
        panel_efficiency=0.2,
    )


def test_require_columns_raises() -> None:
    df = pd.DataFrame({"timestamp_utc": [pd.Timestamp("2026-01-01T00:00:00Z")]})
    with pytest.raises(ValueError, match="Missing required columns"):
        integ.hourly_weather_df_to_pv_telemetry(df)


def test_hourly_weather_df_to_pv_telemetry_converts_w_to_kw_and_handles_nan() -> None:
    df = pd.DataFrame(
        {
            "timestamp_utc": [
                pd.Timestamp("2026-01-01T00:00:00Z"),
                pd.Timestamp("2026-01-01T01:00:00Z"),
            ],
            # W/m^2 -> kW/m^2
            "shortwave_radiation": [1000.0, float("nan")],
        }
    )

    telem = integ.hourly_weather_df_to_pv_telemetry(df)

    assert len(telem) == 2
    assert telem[0].solar_radiance_kw_per_m2 == pytest.approx(1.0)
    assert telem[1].solar_radiance_kw_per_m2 is None
    assert telem[0].generation_kw is None  # by design


def test_pv_states_from_hourly_weather_df_estimates_energy() -> None:
    spec = _mk_spec()

    df = pd.DataFrame(
        {
            "timestamp_utc": [pd.Timestamp("2026-01-01T00:00:00Z")],
            "shortwave_radiation": [1000.0],  # 1.0 kW/m^2
        }
    )

    out = integ.pv_states_from_hourly_weather_df(spec, df, timestep_minutes=60)

    assert len(out) == 1
    # energy = radiance(1.0) * area(10) * eff(0.2) * 1h = 2.0 kWh
    assert out[0].energy_kwh == pytest.approx(2.0)
    assert "estimated_from_radiance" in out[0].quality_flags


def test_fetch_live_pv_states_from_openmeteo_uses_weather_fetcher(monkeypatch) -> None:
    spec = _mk_spec()
    called = {"fetch": 0, "to_df": 0}

    class FakeClient:
        pass

    def fake_fetch_forecast(client, params):
        called["fetch"] += 1
        assert isinstance(client, FakeClient)
        assert params["hourly"] == ["shortwave_radiation"]
        return object()

    def fake_to_hourly_df(api_resp, hourly_vars):
        called["to_df"] += 1
        assert hourly_vars == ["shortwave_radiation"]
        return pd.DataFrame(
            {
                "timestamp_utc": [pd.Timestamp("2026-01-01T00:00:00Z")],
                "shortwave_radiation": [1000.0],
            }
        )

    monkeypatch.setattr(integ.weather_fetcher, "fetch_forecast", fake_fetch_forecast)
    monkeypatch.setattr(integ.weather_fetcher, "to_hourly_df", fake_to_hourly_df)

    params = {"hourly": ["shortwave_radiation"]}
    out = integ.fetch_live_pv_states_from_openmeteo(
        spec,
        client=FakeClient(),
        params=params,
        timestep_minutes=60,
    )

    assert called["fetch"] == 1
    assert called["to_df"] == 1
    assert len(out) == 1
    assert out[0].energy_kwh == pytest.approx(2.0)
