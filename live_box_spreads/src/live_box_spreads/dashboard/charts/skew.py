"""Lend vs Borrow rate chart factory."""
from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl
import plotly.graph_objects as go

PLOT_BG = "#1a1a2e"
GRID_COLOR = "#2d2d44"


def make_lend_borrow_chart(df: pl.DataFrame, expiry: Optional[str] = None) -> go.Figure:
    """Scatter plot showing lend vs borrow rates across strikes.

    Lend rate (long box, net debit) is always <= borrow rate (short box, net credit).
    The gap between them is the market-maker's edge.
    """
    layout = dict(
        template="plotly_dark",
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="'Inter', sans-serif", size=11, color="#c0c0d0"),
        margin=dict(l=50, r=20, t=35, b=45),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False, title="Mid Strike"),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, title="Rate (%)"),
        title="Lend vs Borrow Rate",
    )

    if df.is_empty():
        return go.Figure(layout=layout)
    if expiry:
        df = df.filter(pl.col("expiry") == expiry)
    if df.is_empty():
        return go.Figure(layout=layout)

    # Drop rows where either rate is NaN
    df = df.filter(
        pl.col("lend_rate").is_not_nan() & pl.col("borrow_rate").is_not_nan()
    )
    if df.is_empty():
        return go.Figure(layout=layout)

    x = df["mid_strike"].to_numpy()
    lend = (df["lend_rate"] * 100).to_numpy()       # positive (above zero)
    borrow = (df["borrow_rate"] * 100).to_numpy()    # negative (below zero)
    spread = lend - borrow  # lend(+) minus borrow(-) = total width, always > 0

    fig = go.Figure()

    # Zero line for reference
    fig.add_hline(y=0, line_dash="dot", line_color="#444466", line_width=1)

    fig.add_trace(go.Scatter(
        x=x, y=lend,
        mode="markers",
        name="Lend Rate (+)",
        marker=dict(size=5, color="#00d4aa", opacity=0.6),
        hovertemplate="Strike: %{x:,.0f}<br>Lend: +%{y:.2f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=borrow,
        mode="markers",
        name="Borrow Rate (\u2212)",
        marker=dict(size=5, color="#ff6b6b", opacity=0.6),
        hovertemplate="Strike: %{x:,.0f}<br>Borrow: %{y:.2f}%<extra></extra>",
    ))

    title = "Lend vs Borrow Rate"
    if expiry:
        title += f"  ({expiry})"
    layout["title"] = title
    fig.update_layout(**layout)
    return fig
