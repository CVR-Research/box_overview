"""WebSocket-backed streaming data source.

Unifies the old StreamingSnapshotCollector (ingest.py) and
StreamDataSource (dashboard/app.py) into a single implementation.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from threading import Thread
from typing import Dict, List, Optional

import polars as pl

from live_box_spreads.config import Config
from live_box_spreads.engine.spread_builder import SpreadBuilder
from live_box_spreads.providers.alpaca_stream import (
    AlpacaStreamRunner,
    StreamQuoteCache,
)
from live_box_spreads.providers.protocol import DataProvider

LOGGER = logging.getLogger(__name__)


class StreamSource:
    """DataSource backed by a live WebSocket stream.

    On startup, bootstraps from REST (one-time initial fetch), then
    subscribes to WebSocket for live updates. Uses a version counter
    to avoid rebuilding spreads if nothing changed.
    """

    def __init__(
        self,
        provider: DataProvider,
        config: Config,
        api_key: str,
        api_secret: str,
    ) -> None:
        self.provider = provider
        self.config = config
        self.builder = SpreadBuilder(config)
        self._api_key = api_key
        self._api_secret = api_secret
        self.cache = StreamQuoteCache()
        self._runner: Optional[AlpacaStreamRunner] = None
        self._bootstrap_thread: Optional[Thread] = None
        self._last_version = -1
        self._last_frame = pl.DataFrame()
        self._last_update: Optional[datetime] = None
        self._expiries: Dict[str, List[str]] = {}
        self._started = False
        self._bootstrapping = False
        self._last_error: Optional[str] = None
        self._subscribed_symbols = 0
        self._bootstrap_total = 0
        self._bootstrap_done = 0

    def latest_spreads(self) -> pl.DataFrame:
        if not self._started:
            self.start_async()
            return pl.DataFrame()
        version = self.cache.version()
        if version == self._last_version:
            return self._last_frame
        snapshot_time = datetime.now(timezone.utc)
        frames: List[pl.DataFrame] = []
        for ticker, expiries in self._expiries.items():
            spot = self.provider.get_underlying_price(ticker)
            for expiry in expiries:
                quotes = self.cache.snapshot(ticker, expiry)
                if not quotes:
                    continue
                spreads = self.builder.build_from_quotes(
                    ticker, expiry, spot, quotes, snapshot_time
                )
                if not spreads.is_empty():
                    frames.append(spreads)
        if frames:
            df = pl.concat(frames, how="vertical")
        else:
            df = pl.DataFrame()
        self._last_frame = df
        self._last_version = version
        self._last_update = snapshot_time
        return df

    def status(self) -> dict:
        return {
            "source": "stream" if self._started else "stream (init)",
            "version": self.cache.version(),
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "bootstrapping": self._bootstrapping,
            "symbols": self._subscribed_symbols,
            "error": self._last_error,
            "bootstrap_done": self._bootstrap_done,
            "bootstrap_total": self._bootstrap_total,
        }

    def start(self) -> None:
        """Bootstrap from REST, then start the WebSocket stream. Blocks during bootstrap."""
        if self._started:
            return
        self._bootstrapping = True
        self._last_error = None
        self._bootstrap_total = len(self.config.tickers)
        self._bootstrap_done = 0

        symbols: List[str] = []
        expiries: Dict[str, List[str]] = {}

        for ticker in self.config.tickers:
            try:
                chain = self.provider.get_option_chain(ticker)
                if not chain:
                    self._last_error = f"Empty option chain for {ticker}"
                    LOGGER.warning("Stream bootstrap: empty option chain for %s", ticker)
                    self._bootstrap_done += 1
                    continue
                chain_expiries = self.provider.extract_expiries(
                    chain, max_expiries=self.config.expiries_per_ticker
                )
            except Exception:
                self._last_error = f"Failed to fetch option chain for {ticker}"
                LOGGER.exception(
                    "Stream bootstrap: failed to fetch option chain for %s", ticker
                )
                self._bootstrap_done += 1
                continue

            expiries[ticker] = chain_expiries
            LOGGER.info("Stream bootstrap: %s expiries=%d", ticker, len(chain_expiries))
            quotes = self.provider.quotes_from_chain(ticker, chain)
            for quote in quotes:
                if quote.symbol:
                    symbols.append(quote.symbol)
                    self.cache.update(quote)
            self._bootstrap_done += 1

        symbols = sorted(set(symbols))
        self._expiries = expiries

        if symbols:
            self._runner = AlpacaStreamRunner(
                self._api_key, self._api_secret, symbols, self.cache
            )
            self._runner.start()
            self._subscribed_symbols = len(symbols)
            LOGGER.info("Stream bootstrap: subscribed option symbols=%d", len(symbols))
        else:
            self._subscribed_symbols = 0
            self._last_error = self._last_error or "No option symbols found to subscribe"
            LOGGER.warning("Stream bootstrap: no option symbols found")

        self._last_update = datetime.now(timezone.utc)
        self._started = True
        self._bootstrapping = False

    def start_async(self) -> None:
        """Start bootstrap in a background thread (non-blocking)."""
        if self._started or self._bootstrapping:
            return
        self._bootstrapping = True
        self._bootstrap_thread = Thread(target=self.start, daemon=True)
        self._bootstrap_thread.start()
