# price_api_hist.py
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

from .price_api import (
    fetch_inddem_demand,
    fetch_indo_initial_demand,
    fetch_itsdo_demand,
    fetch_mid_price,
    make_session,
    to_dt_utc,
)


@dataclass
class HistoryConfig:
    """Runtime configuration for building price_data.csv historical dataset.

    Attributes:
        base_url: BMRS base URL.
        start_utc: ISO-8601 start datetime (inclusive).
        end_utc: ISO-8601 end datetime (exclusive).
        out_dir: Output directory.
        output_csv: Output CSV filename.

        freq: Master dataset frequency (must be "15min" for your use-case).
        mid_chunk_days: Chunk size in days for MID (price) requests.
        demand_chunk_days: Chunk size in days for demand requests.
        mid_providers: MID provider list to pass to fetch_mid_price.
        timeout: HTTP timeout (seconds).
        continue_on_error: Continue when a chunk fails.

    Notes:
        - This implementation fetches each endpoint separately and joins them
          onto a common 15-minute index (like the original pipeline).
        - For half-hourly series, we resample to 15 minutes with ffill(limit=1),
          which fills the intermediate 15-min slot only.
    """

    base_url: str
    start_utc: str
    end_utc: str

    out_dir: str = "price_output"
    output_csv: str = "price_data.csv"

    freq: str = "15min"

    mid_chunk_days: int = 7
    demand_chunk_days: int = 1

    mid_providers: list[str] | None = None
    timeout: int = 60
    continue_on_error: bool = True


class PriceHistoryBuilder:
    """Build historical dataset by fetching endpoints separately and then joining."""

    def __init__(self, cfg: HistoryConfig) -> None:
        """Initialize builder.

        Args:
            cfg: History configuration.
        """
        self.cfg: HistoryConfig = cfg
        if self.cfg.mid_providers is None:
            self.cfg.mid_providers = []

        self.start_dt: datetime = to_dt_utc(self.cfg.start_utc)
        self.end_dt: datetime = to_dt_utc(self.cfg.end_utc)

        os.makedirs(self.cfg.out_dir, exist_ok=True)

        self.master_index: pd.DatetimeIndex = pd.date_range(
            start=self.start_dt,
            end=self.end_dt,
            freq=self.cfg.freq,
            inclusive="left",
            tz="UTC",
        )

    def _safe_call(
        self,
        fn: Callable[..., pd.DataFrame],
        *args: object,
        **kwargs: object,
    ) -> pd.DataFrame:
        """Call API function with optional error tolerance.

        Args:
            fn: Function that returns a DataFrame.
            *args: Positional args.
            **kwargs: Keyword args.

        Returns:
            pd.DataFrame: Result or empty frame if failed and continue_on_error=True.
        """
        try:
            return fn(*args, **kwargs)
        except Exception as ex:  # noqa: BLE001
            if not self.cfg.continue_on_error:
                raise
            print(f"[WARN] Call failed: {fn.__name__} | {ex}")
            return pd.DataFrame()

    def _fetch_in_chunks(
        self,
        session: object,
        chunk_days: int,
        fetch_fn: Callable[..., pd.DataFrame],
        fetch_args_builder: Callable[[datetime, datetime], tuple[tuple[object, ...], dict]],
        label: str,
    ) -> pd.DataFrame:
        """Fetch an endpoint over [start_dt, end_dt) by chunking time range.

        Args:
            session: requests.Session.
            chunk_days: Chunk size (days).
            fetch_fn: Endpoint fetch function.
            fetch_args_builder: Function building args/kwargs for each chunk.
            label: Friendly label for logging.

        Returns:
            pd.DataFrame: Concatenated endpoint output (may be empty).
        """
        frames: list[pd.DataFrame] = []
        cur: datetime = self.start_dt
        step: timedelta = timedelta(days=chunk_days)
        i: int = 0

        while cur < self.end_dt:
            nxt: datetime = min(cur + step, self.end_dt)

            args, kwargs = fetch_args_builder(cur, nxt)
            df = self._safe_call(fetch_fn, *args, **kwargs)

            if not df.empty:
                frames.append(df)

            if i % 10 == 0:
                print(f"[{label}] {cur.isoformat()} -> {nxt.isoformat()} | got_rows={len(df):,}")

            cur = nxt
            i += 1

        if not frames:
            return pd.DataFrame()

        out = pd.concat(frames, ignore_index=True)
        out = out.loc[:, ~out.columns.duplicated()]
        return out

    @staticmethod
    def _to_series_frame(
        df: pd.DataFrame,
        value_col: str,
        out_col: str,
    ) -> pd.DataFrame:
        """Convert endpoint frame [ts_utc, value] to time-indexed frame.

        Args:
            df: DataFrame with at least ts_utc and value_col.
            value_col: Column containing numeric values.
            out_col: Output column name.

        Returns:
            pd.DataFrame: Indexed by ts_utc with one column out_col.
        """
        if df.empty or "ts_utc" not in df.columns:
            return pd.DataFrame()

        w = df.copy()
        w["ts_utc"] = pd.to_datetime(w["ts_utc"], utc=True, errors="coerce")
        w = w.dropna(subset=["ts_utc"])
        if w.empty:
            return pd.DataFrame()

        w[value_col] = pd.to_numeric(w.get(value_col), errors="coerce")

        # collapse duplicates per timestamp
        w = w.sort_values("ts_utc").drop_duplicates(subset=["ts_utc"], keep="last")
        w = w.set_index("ts_utc")[[value_col]].rename(columns={value_col: out_col})

        # ensure unique index
        w = w[~w.index.duplicated(keep="last")]
        return w

    def _resample_half_hour_to_15m(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Resample half-hourly -> 15m by forward fill one step.

        Args:
            frame: Time-indexed frame.

        Returns:
            pd.DataFrame: Reindexed to master 15m grid.
        """
        if frame.empty:
            return frame

        x = frame.sort_index()
        # For half-hourly sources: fill intermediate 15-min slot only
        x = x.resample("15min").ffill(limit=1)
        x = x.reindex(self.master_index)
        return x

    def build(self) -> pd.DataFrame:
        """Build the historical dataset and save price_data.csv.

        Returns:
            pd.DataFrame: Final dataset with columns:
                timestamp, price, demand_itsdo, demand_indo, demand_inddem
        """
        session = make_session()
        try:
            # -------------------------
            # 1) Fetch each endpoint separately (like the original)
            # -------------------------
            mid_raw = self._fetch_in_chunks(
                session=session,
                chunk_days=self.cfg.mid_chunk_days,
                fetch_fn=fetch_mid_price,
                fetch_args_builder=lambda t0, t1: (
                    (session, self.cfg.base_url, t0, t1, self.cfg.mid_providers, self.cfg.timeout),
                    {},
                ),
                label="MID",
            )

            itsdo_raw = self._fetch_in_chunks(
                session=session,
                chunk_days=self.cfg.demand_chunk_days,
                fetch_fn=fetch_itsdo_demand,
                fetch_args_builder=lambda t0, t1: (
                    (session, self.cfg.base_url, t0, t1, self.cfg.timeout),
                    {},
                ),
                label="ITSDO",
            )

            indo_raw = self._fetch_in_chunks(
                session=session,
                chunk_days=self.cfg.demand_chunk_days,
                fetch_fn=fetch_indo_initial_demand,
                fetch_args_builder=lambda t0, t1: (
                    (session, self.cfg.base_url, t0, t1, self.cfg.timeout),
                    {},
                ),
                label="INDO",
            )

            inddem_raw = self._fetch_in_chunks(
                session=session,
                chunk_days=self.cfg.demand_chunk_days,
                fetch_fn=fetch_inddem_demand,
                fetch_args_builder=lambda t0, t1: (
                    (session, self.cfg.base_url, t0, t1, self.cfg.timeout),
                    {},
                ),
                label="INDDEM",
            )

            # -------------------------
            # 2) Convert to time series frames (index=ts_utc)
            # -------------------------
            mid_ts = self._to_series_frame(mid_raw, value_col="price", out_col="price")
            itsdo_ts = self._to_series_frame(
                itsdo_raw, value_col="demand_itsdo", out_col="demand_itsdo"
            )
            indo_ts = self._to_series_frame(
                indo_raw, value_col="demand_indo", out_col="demand_indo"
            )
            inddem_ts = self._to_series_frame(
                inddem_raw, value_col="demand_inddem", out_col="demand_inddem"
            )

            print(f"[MID] ts_rows={len(mid_ts):,}")
            print(f"[ITSDO] ts_rows={len(itsdo_ts):,}")
            print(f"[INDO] ts_rows={len(indo_ts):,}")
            print(f"[INDDEM] ts_rows={len(inddem_ts):,}")

            # -------------------------
            # 3) Reindex/resample to master 15m grid
            # -------------------------
            # MID is typically half-hourly too; if your MID is already 15m you can swap to .reindex only
            mid_15m = self._resample_half_hour_to_15m(mid_ts)
            itsdo_15m = self._resample_half_hour_to_15m(itsdo_ts)
            indo_15m = self._resample_half_hour_to_15m(indo_ts)
            inddem_15m = self._resample_half_hour_to_15m(inddem_ts)

            # -------------------------
            # 4) Join everything into one dataset
            # -------------------------
            dataset = pd.DataFrame(index=self.master_index)
            dataset.index.name = "timestamp"

            dataset = dataset.join(mid_15m, how="left")
            dataset = dataset.join(itsdo_15m, how="left")
            dataset = dataset.join(indo_15m, how="left")
            dataset = dataset.join(inddem_15m, how="left")

            out = dataset.reset_index()

            # Ensure column order
            wanted = ["timestamp", "price", "demand_itsdo", "demand_indo", "demand_inddem"]
            for c in wanted:
                if c not in out.columns:
                    out[c] = pd.NA
            out = out[wanted]

            # -------------------------
            # 5) Save
            # -------------------------
            out_path = os.path.join(self.cfg.out_dir, self.cfg.output_csv)
            out.to_csv(out_path, index=False)
            print(f"[SAVE] {out_path} | rows={len(out):,} cols={out.shape[1]}")

            return out

        finally:
            session.close()


if __name__ == "__main__":
    cfg = HistoryConfig(
        base_url="https://data.elexon.co.uk/bmrs/api/v1",
        start_utc="2025-06-01T00:00:00Z",
        end_utc="2025-06-03T00:00:00Z",
        out_dir="price_output",
        output_csv="price_data.csv",
        mid_providers=["APXMIDP"],
        mid_chunk_days=7,
        demand_chunk_days=1,
        timeout=60,
        continue_on_error=True,
    )

    PriceHistoryBuilder(cfg).build()
