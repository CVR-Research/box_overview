from datetime import datetime, timedelta, timezone
import os
from pathlib import Path

import polars as pl

from live_box_spreads.sources.snapshot_source import SnapshotSource


def _write_snapshot(path: Path, snapshot_time: datetime, ticker: str) -> None:
    frame = pl.DataFrame({
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
    })
    frame.write_parquet(path)


def test_load_filters_by_cutoff(tmp_path: Path):
    now = datetime.now(timezone.utc)
    recent_path = tmp_path / "snapshot_recent.parquet"
    old_path = tmp_path / "snapshot_old.parquet"
    _write_snapshot(recent_path, now, "SPX")
    _write_snapshot(old_path, now - timedelta(hours=2), "NDX")

    old_ts = (now - timedelta(hours=2)).timestamp()
    recent_ts = now.timestamp()
    os.utime(old_path, (old_ts, old_ts))
    os.utime(recent_path, (recent_ts, recent_ts))

    source = SnapshotSource(tmp_path, history_minutes=30)
    df = source.latest_spreads()
    assert not df.is_empty()
    assert set(df["ticker"].unique().to_list()) == {"SPX"}


def test_status(tmp_path: Path):
    source = SnapshotSource(tmp_path, history_minutes=30)
    status = source.status()
    assert status["source"] == "snapshot"
    assert status["file_count"] == 0
