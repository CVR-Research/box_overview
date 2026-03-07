"""Reads box spread snapshots from parquet/csv files on disk."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

import polars as pl

LOGGER = logging.getLogger(__name__)


class SnapshotSource:
    """DataSource that reads historical snapshots from disk."""

    def __init__(self, snapshot_dir: Path, history_minutes: int = 30) -> None:
        self.snapshot_dir = snapshot_dir
        self.history_minutes = history_minutes

    def latest_spreads(self) -> pl.DataFrame:
        if not self.snapshot_dir.exists():
            return pl.DataFrame()
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=self.history_minutes)
        parquet_files = _list_snapshot_files(self.snapshot_dir, "parquet")
        if parquet_files:
            selected = _select_recent_files(parquet_files, cutoff.timestamp())
            frames = [pl.read_parquet(path) for path in selected]
        else:
            csv_files = _list_snapshot_files(self.snapshot_dir, "csv")
            if not csv_files:
                return pl.DataFrame()
            selected = _select_recent_files(csv_files, cutoff.timestamp())
            frames = [pl.read_csv(path) for path in selected]
        if not frames:
            return pl.DataFrame()
        return pl.concat(frames, how="vertical")

    def status(self) -> dict:
        parquet_files = _list_snapshot_files(self.snapshot_dir, "parquet")
        files = parquet_files or _list_snapshot_files(self.snapshot_dir, "csv")
        return {
            "source": "snapshot",
            "file_count": len(files),
            "latest_mtime": files[0][1] if files else None,
        }


def _list_snapshot_files(
    snapshot_dir: Path, suffix: str
) -> List[Tuple[Path, float]]:
    """List snapshot files sorted by mtime descending (newest first)."""
    files: List[Tuple[Path, float]] = []
    for path in snapshot_dir.glob(f"snapshot_*.{suffix}"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        files.append((path, mtime))
    files.sort(key=lambda item: item[1], reverse=True)
    return files


def _select_recent_files(
    files: List[Tuple[Path, float]], cutoff_ts: float
) -> List[Path]:
    """Select files newer than the cutoff timestamp. Always includes at least one."""
    selected: List[Path] = []
    for path, mtime in files:
        if mtime < cutoff_ts and selected:
            break
        selected.append(path)
    return selected
