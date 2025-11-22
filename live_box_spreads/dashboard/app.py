from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import polars as pl
import yaml
from dash import Dash, Input, Output, State, dash_table, dcc, html
import plotly.graph_objects as go

BASE_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = BASE_DIR / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
    CONFIG = yaml.safe_load(fh) or {}

SNAPSHOT_DIR = (BASE_DIR / CONFIG.get("snapshot_storage", "data/live_snapshots")).resolve()
DEFAULT_HISTORY_MINUTES = CONFIG.get("surface_history_minutes", 30)

app = Dash(__name__)
app.title = "Live Box Spreads"


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_recent_snapshots(history_minutes: int) -> pd.DataFrame:
    if not SNAPSHOT_DIR.exists():
        return pd.DataFrame()
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=history_minutes)
    frames: List[pl.DataFrame] = []
    parquet_files = sorted(SNAPSHOT_DIR.glob("snapshot_*.parquet"), key=lambda p: p.stat().st_mtime, reverse=True)
    for path in parquet_files:
        if path.stat().st_mtime < cutoff.timestamp() and frames:
            break
        frames.append(pl.read_parquet(path))
    if not frames:
        csv_files = sorted(SNAPSHOT_DIR.glob("snapshot_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in csv_files:
            if path.stat().st_mtime < cutoff.timestamp() and frames:
                break
            frames.append(pl.read_csv(path))
    if not frames:
        return pd.DataFrame()
    df_pl = pl.concat(frames, how="vertical").with_columns(
        pl.col("snapshot_time").str.strptime(pl.Datetime, strict=False, utc=True).alias("snapshot_time_dt")
    )
    df = df_pl.to_pandas()
    df["expiry_dt"] = pd.to_datetime(df["expiry"], format="%Y-%m-%d", utc=True, errors="coerce")
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
    df = df.sort_values("mid_rate", ascending=False)
    top_lend = df[df["mid_rate"] > 0].nlargest(5, "mid_rate")[cols]
    top_borrow = df[df["mid_rate"] < 0].nsmallest(5, "mid_rate")[cols]
    combined = pd.concat([top_lend, top_borrow])
    combined["mid_rate"] = combined["mid_rate"].map(lambda v: f"{v*100:.2f}%")
    combined["bid_rate"] = combined["bid_rate"].map(lambda v: f"{v*100:.2f}%")
    combined["ask_rate"] = combined["ask_rate"].map(lambda v: f"{v*100:.2f}%")
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
        dcc.Interval(id="refresh-interval", interval=CONFIG.get("update_interval_seconds", 60) * 1000, n_intervals=0),
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
    Input("refresh-interval", "n_intervals"),
    Input("history-slider", "value"),
)
def refresh_snapshots(_, history_minutes):
    history_minutes = history_minutes or DEFAULT_HISTORY_MINUTES
    df = load_recent_snapshots(history_minutes)
    return df.to_dict("records") if not df.empty else []


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
    df = pd.DataFrame(data)
    if df.empty:
        return [], None, [], None
    df = df[df["ticker"] == ticker]
    expiries = sorted(df["expiry"].unique())
    options = [{"label": e, "value": e} for e in expiries]
    expiry_value = expiry_value if expiry_value in expiries else (expiries[0] if expiries else None)
    skew_value = skew_value if skew_value in expiries else expiry_value
    return options, expiry_value, options, skew_value


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
    df = pd.DataFrame(data)
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
