"""Rate curve and heatmap chart factories."""
from __future__ import annotations

import numpy as np
import polars as pl
import plotly.graph_objects as go

DARK_TEMPLATE = "plotly_dark"
PLOT_BG = "#1a1a2e"
GRID_COLOR = "#2d2d44"
ACCENT = "#00d4aa"


def _base_layout(**overrides) -> dict:
    """Shared layout settings for all charts."""
    defaults = dict(
        template=DARK_TEMPLATE,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="'Inter', sans-serif", size=11, color="#c0c0d0"),
        margin=dict(l=50, r=20, t=35, b=45),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
    )
    defaults.update(overrides)
    return defaults


def make_rate_curve(df: pl.DataFrame, expiry: str | None = None) -> go.Figure:
    """Scatter plot of mid_strike vs mid_rate for a given expiry, colored by width."""
    if df.is_empty():
        return go.Figure(layout=_base_layout(title="Implied Rate vs Strike"))

    if expiry:
        df = df.filter(pl.col("expiry") == expiry)
    if df.is_empty():
        return go.Figure(layout=_base_layout(title="Implied Rate vs Strike"))

    x = df["mid_strike"].to_numpy()
    y = (df["mid_rate"] * 100).to_numpy()  # Convert to %
    widths = df["width"].to_numpy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode="markers",
        marker=dict(
            size=np.clip(widths / 8, 3, 18),
            color=widths,
            colorscale="Viridis",
            colorbar=dict(title="Width", thickness=12, len=0.6),
            opacity=0.7,
            line=dict(width=0),
        ),
        hovertemplate="Strike: %{x:,.0f}<br>Rate: %{y:.2f}%<br><extra></extra>",
    ))

    title = f"Implied Rate vs Strike"
    if expiry:
        title += f"  ({expiry})"

    fig.update_layout(**_base_layout(
        title=title,
        xaxis_title="Mid Strike",
        yaxis_title="Implied Rate (%)",
    ))
    return fig


def make_rate_heatmap(df: pl.DataFrame, expiry: str | None = None) -> go.Figure:
    """Heatmap of strike pairs (kl x ks) colored by mid_rate."""
    if df.is_empty():
        return go.Figure(layout=_base_layout(title="Rate Heatmap"))

    if expiry:
        df = df.filter(pl.col("expiry") == expiry)
    if df.is_empty():
        return go.Figure(layout=_base_layout(title="Rate Heatmap"))

    # Aggregate to get one rate per (kl, ks) pair
    agg = (
        df.group_by(["kl", "ks"])
        .agg(pl.col("mid_rate").mean().alias("rate"))
        .sort(["kl", "ks"])
    )

    # Sample down if too many pairs (heatmap gets unreadable)
    if agg.height > 2000:
        agg = agg.sample(n=2000, seed=42)

    kl_vals = sorted(agg["kl"].unique().to_list())
    ks_vals = sorted(agg["ks"].unique().to_list())

    if len(kl_vals) < 2 or len(ks_vals) < 2:
        return go.Figure(layout=_base_layout(title="Rate Heatmap (insufficient data)"))

    kl_idx = {v: i for i, v in enumerate(kl_vals)}
    ks_idx = {v: i for i, v in enumerate(ks_vals)}

    z = np.full((len(ks_vals), len(kl_vals)), np.nan)
    for row in agg.iter_rows(named=True):
        i = ks_idx[row["ks"]]
        j = kl_idx[row["kl"]]
        z[i, j] = row["rate"] * 100  # Convert to %

    fig = go.Figure(data=go.Heatmap(
        x=[f"{v:,.0f}" for v in kl_vals],
        y=[f"{v:,.0f}" for v in ks_vals],
        z=z,
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Rate %", thickness=12, len=0.6),
        hovertemplate="Lower K: %{x}<br>Upper K: %{y}<br>Rate: %{z:.2f}%<extra></extra>",
    ))

    title = "Rate Heatmap (Strike Pairs)"
    if expiry:
        title += f"  ({expiry})"

    fig.update_layout(**_base_layout(
        title=title,
        xaxis_title="Lower Strike (KL)",
        yaxis_title="Upper Strike (KS)",
    ))
    # Override axis for heatmap
    fig.update_xaxes(
        type="category",
        tickangle=45,
        nticks=20,
    )
    fig.update_yaxes(
        type="category",
        nticks=20,
    )
    return fig
