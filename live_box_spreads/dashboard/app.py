from __future__ import annotations
from collections import OrderedDict
from functools import lru_cache
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from threading import Lock
from typing import List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
import polars as pl
import yaml
from dash import Dash, Input, Output, State, dash_table, dcc, html
import plotly.graph_objects as go

from alpaca_client import AlpacaMarketDataClient
from alpaca_stream import AlpacaStreamRunner, StreamQuoteCache
from ingest import Config as IngestConfig
from ingest import build_spreads, pivot_calls_puts, quotes_to_frame

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config.yaml"
CV_PATH = BASE_DIR / "cv.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
    CONFIG = yaml.safe_load(fh) or {}
if CV_PATH.exists():
    with open(CV_PATH, "r", encoding="utf-8") as fh:
        CV = yaml.safe_load(fh) or {}
else:
    CV = {}

SNAPSHOT_DIR = (BASE_DIR / CONFIG.get("snapshot_storage", "data/live_snapshots")).resolve()
DEFAULT_HISTORY_MINUTES = CONFIG.get("surface_history_minutes", 30)
USE_STREAM = os.environ.get("DASH_USE_STREAM", "0") == "1"
UPDATE_INTERVAL_MS = int(
    os.environ.get(
        "DASH_UPDATE_MS",
        2000 if USE_STREAM else CONFIG.get("update_interval_seconds", 60) * 1000,
    )
)

app = Dash(__name__)
app.title = "Live Box Spreads"
MAX_CACHE_ITEMS = 6


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

class SnapshotCache:
    def __init__(self, max_items: int = MAX_CACHE_ITEMS) -> None:
        self._lock = Lock()
        self._data: OrderedDict[str, pd.DataFrame] = OrderedDict()
        self._max_items = max_items

    def put(self, df: pd.DataFrame) -> str:
        key = uuid4().hex
        with self._lock:
            self._data[key] = df
            self._data.move_to_end(key)
            while len(self._data) > self._max_items:
                self._data.popitem(last=False)
        return key

    def get(self, key: Optional[str]) -> pd.DataFrame:
        if not key:
            return pd.DataFrame()
        with self._lock:
            df = self._data.get(key)
            if df is None:
                return pd.DataFrame()
            self._data.move_to_end(key)
            return df


SNAPSHOT_CACHE = SnapshotCache()


def _alpaca_credentials() -> tuple[str, str]:
    api_key = os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_API_KEY_ID")
    api_secret = os.environ.get("ALPACA_API_SECRET") or os.environ.get("ALPACA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET env vars are required for stream mode")
    return api_key, api_secret


def build_spreads_from_quotes(
    ticker: str,
    expiry: str,
    spot: Optional[float],
    quotes: List,
    config: IngestConfig,
    snapshot_time: datetime,
) -> pl.DataFrame:
    merged = pivot_calls_puts(quotes_to_frame(quotes))
    if merged.is_empty():
        return pl.DataFrame()
    expiry_dt = datetime.fromisoformat(expiry).replace(tzinfo=timezone.utc)
    return build_spreads(ticker, expiry_dt, spot, merged, config, snapshot_time)


class StreamDataSource:
    def __init__(self, config: IngestConfig) -> None:
        self.config = config
        self.cache = StreamQuoteCache()
        self._runner: Optional[AlpacaStreamRunner] = None
        self._last_version = -1
        self._last_frame = pd.DataFrame()
        self._last_update: Optional[datetime] = None
        self._expiries: dict[str, List[str]] = {}
        self._started = False
        self._client: Optional[AlpacaMarketDataClient] = None

    def start(self) -> None:
        if self._started:
            return
        api_key, api_secret = _alpaca_credentials()
        client = AlpacaMarketDataClient(api_key, api_secret)
        self._client = client
        symbols: List[str] = []
        expiries: dict[str, List[str]] = {}
        for ticker in self.config.tickers:
            try:
                chain_expiries = client.get_expiries(ticker)
            except Exception:
                continue
            chain_expiries = [e for e in chain_expiries if e][: self.config.expiries_per_ticker]
            expiries[ticker] = chain_expiries
            for expiry in chain_expiries:
                try:
                    quotes = client.get_option_quotes(ticker, expiry)
                except Exception:
                    continue
                for quote in quotes:
                    if quote.symbol:
                        symbols.append(quote.symbol)
                        self.cache.update(quote)
        symbols = sorted(set(symbols))
        self._expiries = expiries
        self._runner = AlpacaStreamRunner(api_key, api_secret, symbols, self.cache)
        self._runner.start()
        self._last_update = datetime.now(timezone.utc)
        self._started = True

    def latest_frame(self) -> pd.DataFrame:
        if not self._started:
            self.start()
        version = self.cache.version()
        if version == self._last_version:
            return self._last_frame
        snapshot_time = datetime.now(timezone.utc)
        frames: List[pl.DataFrame] = []
        client = self._client
        if client is None:
            return pd.DataFrame()
        for ticker, expiries in self._expiries.items():
            spot = client.get_underlying_price(ticker)
            for expiry in expiries:
                quotes = self.cache.snapshot(ticker, expiry)
                if not quotes:
                    continue
                spreads = build_spreads_from_quotes(
                    ticker,
                    expiry,
                    spot,
                    quotes,
                    self.config,
                    snapshot_time,
                )
                if not spreads.is_empty():
                    frames.append(spreads)
        if frames:
            df_pl = pl.concat(frames, how="vertical")
            df = df_pl.to_pandas()
            df["snapshot_time_dt"] = pd.to_datetime(df["snapshot_time"], utc=True, errors="coerce")
            df["expiry_dt"] = pd.to_datetime(df["expiry"], format="%Y-%m-%d", utc=True, errors="coerce")
        else:
            df = pd.DataFrame()
        self._last_frame = df
        self._last_version = version
        self._last_update = snapshot_time
        return df

    def status(self) -> dict:
        return {
            "source": "stream" if self._started else "stream (init)",
            "version": self.cache.version(),
            "last_update": self._last_update.isoformat() if self._last_update else None,
        }


STREAM_SOURCE = StreamDataSource(IngestConfig.from_dict(CONFIG)) if USE_STREAM else None


def _list_snapshot_files(snapshot_dir: Path, suffix: str) -> List[tuple[Path, float]]:
    files: List[tuple[Path, float]] = []
    for path in snapshot_dir.glob(f"snapshot_*.{suffix}"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        files.append((path, mtime))
    files.sort(key=lambda item: item[1], reverse=True)
    return files


def _select_recent_files(files: List[tuple[Path, float]], cutoff_ts: float) -> List[Path]:
    selected: List[Path] = []
    for path, mtime in files:
        if mtime < cutoff_ts and selected:
            break
        selected.append(path)
    return selected


def load_recent_snapshots_from(snapshot_dir: Path, history_minutes: int) -> pd.DataFrame:
    if not snapshot_dir.exists():
        return pd.DataFrame()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=history_minutes)
    parquet_files = _list_snapshot_files(snapshot_dir, "parquet")
    if parquet_files:
        selected = _select_recent_files(parquet_files, cutoff.timestamp())
        frames = [pl.read_parquet(path) for path in selected]
    else:
        csv_files = _list_snapshot_files(snapshot_dir, "csv")
        if not csv_files:
            return pd.DataFrame()
        selected = _select_recent_files(csv_files, cutoff.timestamp())
        frames = [pl.read_csv(path) for path in selected]
    if not frames:
        return pd.DataFrame()
    df_pl = pl.concat(frames, how="vertical")
    df = df_pl.to_pandas()
    df["snapshot_time_dt"] = pd.to_datetime(df["snapshot_time"], utc=True, errors="coerce")
    df["expiry_dt"] = pd.to_datetime(df["expiry"], format="%Y-%m-%d", utc=True, errors="coerce")
    return df


@lru_cache(maxsize=12)
def _cached_load(
    snapshot_dir: str, history_minutes: int, latest_mtime: float, file_count: int
) -> pd.DataFrame:
    return load_recent_snapshots_from(Path(snapshot_dir), history_minutes)


def load_recent_snapshots(history_minutes: int) -> pd.DataFrame:
    if not SNAPSHOT_DIR.exists():
        return pd.DataFrame()
    parquet_files = _list_snapshot_files(SNAPSHOT_DIR, "parquet")
    files = parquet_files or _list_snapshot_files(SNAPSHOT_DIR, "csv")
    if not files:
        return pd.DataFrame()
    latest_mtime = files[0][1]
    df = _cached_load(str(SNAPSHOT_DIR), history_minutes, latest_mtime, len(files))
    return df


def filter_df(
    df: pd.DataFrame,
    ticker: Optional[str],
    min_volume: Optional[int],
) -> pd.DataFrame:
    if df.empty:
        return df
    if ticker:
        df = df[df["ticker"] == ticker]
    if min_volume:
        df = df[df["min_leg_volume"] >= min_volume]
    return df


def _prepare_surface(df: pd.DataFrame, value_col: str):
    if df.empty or value_col not in df:
        return None
    pivot = (
        df.pivot_table(
            index="snapshot_time_dt",
            columns="mid_strike",
            values=value_col,
            aggfunc="mean",
        )
        .sort_index()
        .sort_index(axis=1)
    )
    if pivot.empty:
        return None
    x_vals = pivot.columns.to_numpy()
    y_vals = pivot.index.to_numpy()
    z_vals = pivot.to_numpy()
    y_offsets = (y_vals - y_vals[0]).astype("timedelta64[s]").astype(float) / 60.0
    x_mesh, y_mesh = np.meshgrid(x_vals, y_offsets)
    tick_text = [ts.strftime("%H:%M:%S") for ts in pivot.index]
    return x_mesh, y_mesh, z_vals, y_offsets, tick_text, value_col


def make_surface(df: pd.DataFrame, value_col: str, title: str) -> go.Figure:
    prepared = _prepare_surface(df, value_col)
    if not prepared:
        return go.Figure()
    x_mesh, y_mesh, z_vals, y_offsets, tick_text, _ = prepared
    fig = go.Figure(
        data=[
            go.Surface(
                x=x_mesh,
                y=y_mesh,
                z=z_vals,
                colorscale="Viridis",
                showscale=True,
                opacity=0.9,
            )
        ]
    )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        title=title,
        scene=dict(
            xaxis_title="Mid Strike",
            yaxis_title="Minutes since first snapshot",
            zaxis_title=title,
            yaxis=dict(tickvals=y_offsets, ticktext=tick_text),
        ),
    )
    return fig


def make_bid_ask_surface(df: pd.DataFrame) -> go.Figure:
    prepared_bid = _prepare_surface(df, "bid_rate")
    prepared_ask = _prepare_surface(df, "ask_rate")
    if not prepared_bid and not prepared_ask:
        return go.Figure()
    fig = go.Figure()
    axis_source = prepared_bid or prepared_ask
    if prepared_bid:
        x_mesh, y_mesh, z_vals, y_offsets, tick_text, _ = prepared_bid
        fig.add_trace(
            go.Surface(
                x=x_mesh,
                y=y_mesh,
                z=z_vals,
                name="Bid",
                colorscale="Blues",
                showscale=False,
                opacity=0.8,
            )
        )
    if prepared_ask:
        x_mesh, y_mesh, z_vals, y_offsets, tick_text, _ = prepared_ask
        fig.add_trace(
            go.Surface(
                x=x_mesh,
                y=y_mesh,
                z=z_vals,
                name="Ask",
                colorscale="Reds",
                showscale=False,
                opacity=0.6,
            )
        )
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        title="Bid vs Ask Implied Rate",
        scene=dict(
            xaxis_title="Mid Strike",
            yaxis_title="Minutes since first snapshot",
            zaxis_title="Rate",
            yaxis=dict(
                tickvals=axis_source[3] if axis_source else [],
                ticktext=axis_source[4] if axis_source else [],
            ),
        ),
    )
    return fig


def make_term_structure(df: pd.DataFrame, target: float, band: float) -> go.Figure:
    if df.empty:
        return go.Figure()
    df = df.dropna(subset=["moneyness_mid", "mid_rate"])
    mask = df["moneyness_mid"].sub(target).abs() <= band
    subset = df[mask]
    if subset.empty:
        return go.Figure()
    grouped = subset.groupby("expiry")["mid_rate"].agg(["mean", "max", "min"]).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=grouped["expiry"], y=grouped["mean"], mode="lines+markers", name="Mean"))
    fig.add_trace(go.Scatter(x=grouped["expiry"], y=grouped["max"], mode="lines", name="Max", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=grouped["expiry"], y=grouped["min"], mode="lines", name="Min", line=dict(dash="dot")))
    fig.update_layout(title="Term Structure (selected moneyness)", xaxis_title="Expiry", yaxis_title="Implied Rate")
    return fig


def make_skew(df: pd.DataFrame, expiry: Optional[str]) -> go.Figure:
    if df.empty:
        return go.Figure()
    if expiry:
        df = df[df["expiry"] == expiry]
    if df.empty:
        return go.Figure()
    fig = go.Figure(
        data=[
            go.Scatter(
                x=df["moneyness_mid"],
                y=df["mid_rate"],
                mode="markers",
                marker=dict(size=np.clip(df["width"], 1, 50), color=df["width"], colorscale="Turbo"),
                text=[f"kl={row.kl:.0f}, ks={row.ks:.0f}" for row in df.itertuples()],
            )
        ]
    )
    fig.update_layout(title="Skew (Mid strike / Spot)", xaxis_title="Moneyness", yaxis_title="Implied Rate")
    return fig


def make_top_table(df: pd.DataFrame) -> List[dict]:
    if df.empty:
        return []
    cols = ["ticker", "expiry", "kl", "ks", "width", "mid_rate", "bid_rate", "ask_rate", "min_leg_volume"]
    top_lend = df[df["mid_rate"] > 0].nlargest(5, "mid_rate")[cols]
    top_borrow = df[df["mid_rate"] < 0].nsmallest(5, "mid_rate")[cols]
    combined = pd.concat([top_lend, top_borrow])
    combined["mid_rate"] = np.char.mod("%.2f%%", combined["mid_rate"].to_numpy() * 100.0)
    combined["bid_rate"] = np.char.mod("%.2f%%", combined["bid_rate"].to_numpy() * 100.0)
    combined["ask_rate"] = np.char.mod("%.2f%%", combined["ask_rate"].to_numpy() * 100.0)
    return combined.to_dict("records")


def monyness_target(mode: str) -> float:
    mapping = {
        "ATM": 1.0,
        "OTM Calls": 1.05,
        "OTM Puts": 0.95,
        "Deep OTM Calls": 1.15,
        "Deep OTM Puts": 0.85,
    }
    return mapping.get(mode, 1.0)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app.layout = html.Div(
    [
        dcc.Store(id="snapshot-store"),
        dcc.Store(id="status-store"),
        dcc.Interval(id="refresh-interval", interval=UPDATE_INTERVAL_MS, n_intervals=0),
        html.Div(
            id="stream-status",
            style={
                "padding": "0.5rem 1rem 1rem 1rem",
                "fontSize": "0.9rem",
                "color": "#444",
                "borderBottom": "1px solid #eee",
                "marginBottom": "1rem",
            },
        ),
        html.Div(
            [
                html.H2("Live Box Spread Monitor"),
                html.Label("Ticker"),
                dcc.Dropdown(id="ticker-selector", options=[{"label": t, "value": t} for t in CONFIG.get("tickers", [])], value=CONFIG.get("tickers", [None])[0]),
                html.Label("History (minutes)"),
                dcc.Slider(id="history-slider", min=5, max=180, step=5, value=DEFAULT_HISTORY_MINUTES, marks={i: str(i) for i in range(10, 190, 30)}),
                html.Label("Min volume per leg"),
                dcc.Input(id="min-volume-input", type="number", value=CONFIG.get("min_volume", 0), min=0, step=10),
                html.Label("Expiry (surface)"),
                dcc.Dropdown(id="expiry-dropdown", multi=False),
                html.Label("Expiry (skew)"),
                dcc.Dropdown(id="skew-expiry-dropdown", multi=False),
                html.Label("Term structure moneyness"),
                dcc.Dropdown(
                    id="moneyness-mode",
                    options=[{"label": k, "value": k} for k in ["ATM", "OTM Calls", "OTM Puts", "Deep OTM Calls", "Deep OTM Puts"]],
                    value="ATM",
                ),
                html.Label("Moneyness band (+/- %)"),
                dcc.Slider(id="moneyness-band", min=0.01, max=0.2, step=0.01, value=0.03, marks={0.05: "5%", 0.1: "10%", 0.2: "20%"}),
                html.Hr(),
                html.H4("Profile"),
                html.P(CV.get("summary", "Systematic researcher focused on options microstructure, rate extraction, and fast data pipelines.")),
                html.H5("Skills"),
                html.Ul([html.Li(skill) for skill in CV.get("skills", ["Options analytics", "Event-driven systems", "Python / Polars", "Latency-sensitive data"])]),
                html.H5("Projects"),
                html.Ul(
                    [
                        html.Li(f"{proj.get('name', '')} - {proj.get('impact', '')}")
                        for proj in CV.get(
                            "projects",
                            [
                                {"name": "Live Box Spread Monitor", "impact": "Streams OPRA quotes, derives implied rates in real time"},
                                {"name": "Options Surface Toolkit", "impact": "Fast skew/term-structure modeling"},
                            ],
                        )
                    ]
                ),
            ],
            className="side-panel",
            style={"width": "20%", "display": "inline-block", "verticalAlign": "top", "padding": "1rem"},
        ),
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id="surface-mid", style={"height": "400px"}),
                        dcc.Graph(id="surface-bidask", style={"height": "400px"}),
                    ]
                ),
                html.Div(
                    [
                        dcc.Graph(id="volume-surface", style={"height": "400px"}),
                        dcc.Graph(id="term-structure", style={"height": "400px"}),
                    ],
                ),
                html.Div(
                    [
                        dcc.Graph(id="skew-chart", style={"height": "400px"}),
                        dash_table.DataTable(
                            id="opportunities-table",
                            columns=[
                                {"name": "Ticker", "id": "ticker"},
                                {"name": "Expiry", "id": "expiry"},
                                {"name": "KL", "id": "kl"},
                                {"name": "KS", "id": "ks"},
                                {"name": "Width", "id": "width"},
                                {"name": "Mid Rate", "id": "mid_rate"},
                                {"name": "Bid Rate", "id": "bid_rate"},
                                {"name": "Ask Rate", "id": "ask_rate"},
                                {"name": "Min Leg Vol", "id": "min_leg_volume"},
                            ],
                            style_table={"height": "350px", "overflowY": "auto"},
                        ),
                    ],
                ),
            ],
            className="main-panel",
            style={"width": "78%", "display": "inline-block", "padding": "1rem"},
        ),
    ]
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@app.callback(
    Output("snapshot-store", "data"),
    Output("status-store", "data"),
    Input("refresh-interval", "n_intervals"),
    Input("history-slider", "value"),
)
def refresh_snapshots(_, history_minutes):
    history_minutes = history_minutes or DEFAULT_HISTORY_MINUTES
    if USE_STREAM and STREAM_SOURCE is not None:
        df = STREAM_SOURCE.latest_frame()
        status = STREAM_SOURCE.status()
    else:
        df = load_recent_snapshots(history_minutes)
        status = {"source": "snapshot"}
    status["rows"] = int(len(df)) if not df.empty else 0
    status["updated_at"] = datetime.now(timezone.utc).isoformat()
    return SNAPSHOT_CACHE.put(df) if not df.empty else "", status


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
    df = SNAPSHOT_CACHE.get(data)
    if df.empty:
        return [], None, [], None
    df = df[df["ticker"] == ticker]
    expiries = sorted(df["expiry"].unique())
    options = [{"label": e, "value": e} for e in expiries]
    expiry_value = expiry_value if expiry_value in expiries else (expiries[0] if expiries else None)
    skew_value = skew_value if skew_value in expiries else expiry_value
    return options, expiry_value, options, skew_value


@app.callback(
    Output("stream-status", "children"),
    Input("status-store", "data"),
)
def update_status(status):
    if not status:
        return "Status: waiting for data..."
    parts = [
        f"Source: {status.get('source', 'snapshot')}",
        f"Rows: {status.get('rows', 0)}",
    ]
    if status.get("version") is not None:
        parts.append(f"Stream version: {status.get('version')}")
    if status.get("last_update"):
        parts.append(f"Stream update: {status.get('last_update')}")
    if status.get("updated_at"):
        parts.append(f"Dashboard refresh: {status.get('updated_at')}")
    return " • ".join(parts)


@app.callback(
    Output("surface-mid", "figure"),
    Output("surface-bidask", "figure"),
    Output("volume-surface", "figure"),
    Output("term-structure", "figure"),
    Output("skew-chart", "figure"),
    Output("opportunities-table", "data"),
    Input("snapshot-store", "data"),
    Input("ticker-selector", "value"),
    Input("expiry-dropdown", "value"),
    Input("skew-expiry-dropdown", "value"),
    Input("min-volume-input", "value"),
    Input("moneyness-mode", "value"),
    Input("moneyness-band", "value"),
)
def update_visuals(
    data,
    ticker,
    expiry_surface,
    expiry_skew,
    min_volume,
    moneyness_mode,
    moneyness_band,
):
    df = SNAPSHOT_CACHE.get(data)
    if df.empty:
        empty_fig = go.Figure()
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, []
    df = filter_df(df, ticker, min_volume)
    if df.empty:
        empty_fig = go.Figure()
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, []
    surface_df = df[df["expiry"] == expiry_surface] if expiry_surface else df
    surface_fig = make_surface(surface_df, "mid_rate", f"Mid Rate Surface {ticker}")
    bid_fig = make_bid_ask_surface(surface_df)
    volume_fig = make_surface(surface_df, "total_leg_volume", "Volume Surface")
    term_fig = make_term_structure(df, monyness_target(moneyness_mode), moneyness_band or 0.03)
    skew_fig = make_skew(df, expiry_skew or expiry_surface)
    table = make_top_table(df)
    return surface_fig, bid_fig, volume_fig, term_fig, skew_fig, table


if __name__ == "__main__":
    app.run_server(debug=True)
