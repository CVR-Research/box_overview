"""Dash callback registrations.

All callbacks capture the DataSource and Config via closure.
No global state.
"""
from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from threading import Lock
from typing import Optional
from uuid import uuid4

import polars as pl
import plotly.graph_objects as go
from dash import Dash, Input, Output, State

from live_box_spreads.config import Config
from live_box_spreads.dashboard.charts.skew import make_lend_borrow_chart
from live_box_spreads.dashboard.charts.surfaces import make_rate_curve, make_rate_heatmap
from live_box_spreads.dashboard.charts.term_structure import make_term_structure
from live_box_spreads.dashboard.table import make_top_table
from live_box_spreads.sources.protocol import DataSource


class ServerSideCache:
    """Thread-safe server-side Polars DataFrame cache."""

    def __init__(self, max_items: int = 6) -> None:
        self._lock = Lock()
        self._data: OrderedDict[str, pl.DataFrame] = OrderedDict()
        self._max_items = max_items

    def put(self, df: pl.DataFrame) -> str:
        key = uuid4().hex
        with self._lock:
            self._data[key] = df.clone()
            self._data.move_to_end(key)
            while len(self._data) > self._max_items:
                self._data.popitem(last=False)
        return key

    def get(self, key: Optional[str]) -> pl.DataFrame:
        if not key:
            return pl.DataFrame()
        with self._lock:
            df = self._data.get(key)
            if df is None:
                return pl.DataFrame()
            self._data.move_to_end(key)
            return df.clone()


def _empty_fig() -> go.Figure:
    """Return a styled empty figure."""
    return go.Figure(layout=dict(
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#c0c0d0"),
    ))


def register_callbacks(app: Dash, source: DataSource, config: Config) -> None:
    """Register all Dash callbacks. source/config captured by closure."""
    cache = ServerSideCache()

    @app.callback(
        Output("snapshot-store", "data"),
        Output("status-store", "data"),
        Input("refresh-interval", "n_intervals"),
        Input("history-slider", "value"),
    )
    def refresh_snapshots(_, history_minutes):
        df = source.latest_spreads()
        status = source.status()
        status["rows"] = df.height if not df.is_empty() else 0
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        return cache.put(df) if not df.is_empty() else "", status

    @app.callback(
        Output("expiry-dropdown", "options"),
        Output("expiry-dropdown", "value"),
        Output("skew-expiry-dropdown", "options"),
        Output("skew-expiry-dropdown", "value"),
        Input("snapshot-store", "data"),
        Input("ticker-selector", "value"),
        State("expiry-dropdown", "value"),
        State("skew-expiry-dropdown", "value"),
    )
    def update_expiry_options(data, ticker, expiry_value, skew_value):
        df = cache.get(data)
        if df.is_empty():
            return [], None, [], None
        if ticker:
            df = df.filter(pl.col("ticker") == ticker)
        expiries = sorted(df["expiry"].unique().to_list())
        options = [{"label": e, "value": e} for e in expiries]
        expiry_value = (
            expiry_value if expiry_value in expiries
            else (expiries[0] if expiries else None)
        )
        skew_value = (
            skew_value if skew_value in expiries
            else expiry_value
        )
        return options, expiry_value, options, skew_value

    @app.callback(
        Output("stream-status", "children"),
        Input("status-store", "data"),
    )
    def update_status(status):
        if not status:
            return "Waiting for data..."
        parts = []
        src = status.get("source", "snapshot")
        rows = status.get("rows", 0)
        parts.append(f"{src.upper()} | {rows:,} rows")
        if status.get("version") is not None:
            parts.append(f"v{status['version']}")
        if status.get("error"):
            parts.append(f"ERR: {status['error']}")
        if status.get("updated_at"):
            try:
                ts = datetime.fromisoformat(status["updated_at"])
                parts.append(ts.strftime("%H:%M:%S UTC"))
            except (ValueError, TypeError):
                pass
        return " | ".join(parts)

    @app.callback(
        Output("rate-curve", "figure"),
        Output("term-structure", "figure"),
        Output("rate-heatmap", "figure"),
        Output("skew-chart", "figure"),
        Output("opportunities-table", "data"),
        Input("snapshot-store", "data"),
        Input("ticker-selector", "value"),
        Input("expiry-dropdown", "value"),
        Input("skew-expiry-dropdown", "value"),
        Input("min-volume-input", "value"),
        Input("width-range", "value"),
        Input("moneyness-band", "value"),
    )
    def update_visuals(data, ticker, expiry, skew_expiry, min_volume, width_range, moneyness_band):
        empty = _empty_fig()
        df = cache.get(data)
        if df.is_empty():
            return empty, empty, empty, empty, []

        # Filter by ticker
        if ticker:
            df = df.filter(pl.col("ticker") == ticker)

        # Filter by min volume
        if min_volume and min_volume > 0 and "min_leg_volume" in df.columns:
            df = df.filter(pl.col("min_leg_volume") >= min_volume)

        # Filter by width range
        if width_range and len(width_range) == 2 and "width" in df.columns:
            df = df.filter(
                (pl.col("width") >= width_range[0]) & (pl.col("width") <= width_range[1])
            )

        # Filter by moneyness band: |mid_strike / spot - 1| <= band
        if moneyness_band and "moneyness_mid" in df.columns:
            band = moneyness_band / 100.0
            df = df.filter(
                (pl.col("moneyness_mid").is_not_nan())
                & ((pl.col("moneyness_mid") - 1.0).abs() <= band)
            )

        if df.is_empty():
            return empty, empty, empty, empty, []

        rate_curve = make_rate_curve(df, expiry)
        term_fig = make_term_structure(df)
        heatmap = make_rate_heatmap(df, expiry)
        bid_ask = make_lend_borrow_chart(df, skew_expiry or expiry)
        table = make_top_table(df)

        return rate_curve, term_fig, heatmap, bid_ask, table
