"""REST-polling data source. Fetches snapshots via the DataProvider and writes to disk."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import polars as pl

from live_box_spreads.config import Config
from live_box_spreads.engine.spread_builder import SpreadBuilder, parse_expiry
from live_box_spreads.engine.transforms import derive_spot_from_chain
from live_box_spreads.providers.protocol import DataProvider

LOGGER = logging.getLogger(__name__)


class RestSource:
    """DataSource that polls a DataProvider for fresh option chains and builds spreads.

    Also writes snapshots to disk and prunes old files.
    """

    def __init__(
        self,
        provider: DataProvider,
        config: Config,
        *,
        storage_dir: Optional[Path] = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.builder = SpreadBuilder(config)
        self.storage_dir = (storage_dir or Path(config.snapshot_storage)).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._last_frame = pl.DataFrame()
        self._last_snapshot_time: Optional[datetime] = None

    def latest_spreads(self) -> pl.DataFrame:
        return self._last_frame

    def status(self) -> dict:
        return {
            "source": "rest",
            "rows": self._last_frame.height,
            "last_snapshot": (
                self._last_snapshot_time.isoformat() if self._last_snapshot_time else None
            ),
        }

    def run_once(self) -> Optional[Path]:
        """Fetch a single snapshot from the provider, build spreads, write to disk."""
        snapshot_time = datetime.now(timezone.utc)
        frames: List[pl.DataFrame] = []

        for ticker in self.config.tickers:
            try:
                chain = self.provider.get_option_chain(ticker)
            except Exception as exc:
                LOGGER.error("Failed to get option chain for %s: %s", ticker, exc)
                continue
            expiries = self.provider.extract_expiries(
                chain, max_expiries=self.config.expiries_per_ticker
            )
            if not expiries:
                continue
            spot = self.provider.get_underlying_price(ticker)
            # Derive spot from options chain if provider can't get it directly
            # (common for indices like SPX, VIX, DJX)
            if spot is None:
                all_quotes = self.provider.quotes_from_chain(ticker, chain)
                spot = derive_spot_from_chain(all_quotes)
                if spot is not None:
                    LOGGER.info("Derived spot for %s from put-call parity: %.2f", ticker, spot)
            for expiry in expiries:
                try:
                    quotes = self.provider.quotes_for_expiry(ticker, chain, expiry)
                except Exception as exc:
                    LOGGER.error("Failed quotes %s %s: %s", ticker, expiry, exc)
                    continue
                spreads = self.builder.build_from_quotes(
                    ticker, expiry, spot, quotes, snapshot_time
                )
                if not spreads.is_empty():
                    frames.append(spreads)

        if not frames:
            LOGGER.warning("No spreads generated in this snapshot")
            return None

        snapshot_df = pl.concat(frames, how="vertical")
        self._last_frame = snapshot_df
        self._last_snapshot_time = snapshot_time
        path = self._write_snapshot(snapshot_df, snapshot_time)
        self._prune_old_snapshots()
        LOGGER.info("Snapshot saved to %s (%d rows)", path.name, snapshot_df.height)
        return path

    def loop(self) -> None:
        """Continuously collect snapshots on an interval."""
        interval = max(self.config.update_interval_seconds, 5)
        while True:
            start = time.time()
            try:
                self.run_once()
            except Exception as exc:
                LOGGER.exception("Snapshot iteration failed: %s", exc)
            elapsed = time.time() - start
            sleep_for = max(interval - elapsed, 0)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _write_snapshot(self, df: pl.DataFrame, ts: datetime) -> Path:
        timestamp = ts.strftime("%Y%m%d_%H%M%S")
        parquet_path = self.storage_dir / f"snapshot_{timestamp}.parquet"
        try:
            df.write_parquet(parquet_path)
            return parquet_path
        except Exception:
            csv_path = parquet_path.with_suffix(".csv")
            df.write_csv(csv_path)
            return csv_path

    def _prune_old_snapshots(self) -> None:
        files = sorted(
            self.storage_dir.glob("snapshot_*"), key=lambda p: p.stat().st_mtime
        )
        excess = len(files) - self.config.max_snapshots_retained
        for path in files[:excess]:
            try:
                path.unlink()
            except OSError:
                continue
