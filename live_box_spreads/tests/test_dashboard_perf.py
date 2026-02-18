from datetime import datetime, timedelta, timezone
import os
from pathlib import Path

import polars as pl

from dashboard import app as dashboard_app


def _write_snapshot(path: Path, snapshot_time: datetime, ticker: str) -> None:
    frame = pl.DataFrame(
        {
            "ticker": [ticker],
            "expiry": ["2026-01-01"],
            "snapshot_time": [snapshot_time.isoformat()],
            "mid_strike": [4000.0],
            "mid_rate": [0.02],
            "bid_rate": [0.019],
            "ask_rate": [0.021],
            "total_leg_volume": [100],
            "moneyness_mid": [1.0],
            "width": [50.0],
            "kl": [4000.0],
            "ks": [4050.0],
            "min_leg_volume": [100],
        }
    )
    frame.write_parquet(path)


def test_load_recent_snapshots_from_filters_by_cutoff(tmp_path: Path):
    now = datetime.now(timezone.utc)
    recent_path = tmp_path / "snapshot_recent.parquet"
    old_path = tmp_path / "snapshot_old.parquet"
    _write_snapshot(recent_path, now, "SPX")
    _write_snapshot(old_path, now - timedelta(hours=2), "NDX")

    old_ts = (now - timedelta(hours=2)).timestamp()
    recent_ts = now.timestamp()
    os.utime(old_path, (old_ts, old_ts))
    os.utime(recent_path, (recent_ts, recent_ts))

    df = dashboard_app.load_recent_snapshots_from(tmp_path, history_minutes=30)
    assert not df.empty
    assert set(df["ticker"].unique()) == {"SPX"}


def test_snapshot_cache_round_trip():
    df = dashboard_app.pd.DataFrame({"ticker": ["SPX"], "expiry": ["2026-01-01"]})
    key = dashboard_app.SNAPSHOT_CACHE.put(df)
    cached = dashboard_app.SNAPSHOT_CACHE.get(key)
    assert cached.equals(df)


def test_make_top_table_formats_rates():
    df = dashboard_app.pd.DataFrame(
        {
            "ticker": ["SPX", "SPX"],
            "expiry": ["2026-01-01", "2026-01-01"],
            "kl": [4000.0, 4010.0],
            "ks": [4050.0, 4060.0],
            "width": [50.0, 50.0],
            "mid_rate": [0.02, -0.01],
            "bid_rate": [0.019, -0.011],
            "ask_rate": [0.021, -0.009],
            "min_leg_volume": [100, 100],
        }
    )
    records = dashboard_app.make_top_table(df)
    assert records
    assert records[0]["mid_rate"].endswith("%")
