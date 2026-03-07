"""Term structure chart factory."""
from __future__ import annotations

import polars as pl
import plotly.graph_objects as go

PLOT_BG = "#1a1a2e"
GRID_COLOR = "#2d2d44"


def make_term_structure(df: pl.DataFrame) -> go.Figure:
    """Rate term structure across expiries — median, mean, and IQR band."""
    layout = dict(
        template="plotly_dark",
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="'Inter', sans-serif", size=11, color="#c0c0d0"),
        margin=dict(l=50, r=20, t=35, b=45),
        xaxis=dict(gridcolor=GRID_COLOR, zeroline=False),
        yaxis=dict(gridcolor=GRID_COLOR, zeroline=False, title="Implied Rate (%)"),
        title="Rate Term Structure",
    )

    if df.is_empty():
        return go.Figure(layout=layout)

    df = df.filter(pl.col("mid_rate").is_not_nan() & pl.col("mid_rate").is_not_null())
    if df.is_empty():
        return go.Figure(layout=layout)

    grouped = (
        df.group_by("expiry")
        .agg(
            pl.col("mid_rate").median().alias("median"),
            pl.col("mid_rate").mean().alias("mean"),
            pl.col("mid_rate").quantile(0.25).alias("q25"),
            pl.col("mid_rate").quantile(0.75).alias("q75"),
            pl.len().alias("n"),
        )
        .sort("expiry")
    )

    if grouped.is_empty():
        return go.Figure(layout=layout)

    expiries = grouped["expiry"].to_list()
    fig = go.Figure()

    q25 = (grouped["q25"] * 100).to_list()
    q75 = (grouped["q75"] * 100).to_list()
    fig.add_trace(go.Scatter(
        x=expiries + expiries[::-1],
        y=q75 + q25[::-1],
        fill="toself",
        fillcolor="rgba(0, 212, 170, 0.12)",
        line=dict(color="rgba(0, 212, 170, 0.3)", width=1),
        name="IQR",
        hoverinfo="skip",
    ))

    medians = (grouped["median"] * 100).to_list()
    fig.add_trace(go.Scatter(
        x=expiries,
        y=medians,
        mode="lines+markers",
        name="Median",
        line=dict(color="#00d4aa", width=2.5),
        marker=dict(size=8, symbol="diamond"),
        hovertemplate="Expiry: %{x}<br>Median: %{y:.2f}%<extra></extra>",
    ))

    means = (grouped["mean"] * 100).to_list()
    fig.add_trace(go.Scatter(
        x=expiries,
        y=means,
        mode="lines+markers",
        name="Mean",
        line=dict(color="#6c8eff", width=2, dash="dash"),
        marker=dict(size=6),
        hovertemplate="Expiry: %{x}<br>Mean: %{y:.2f}%<extra></extra>",
    ))

    counts = grouped["n"].to_list()
    for i, (e, n) in enumerate(zip(expiries, counts)):
        fig.add_annotation(
            x=e, y=medians[i],
            text=f"n={n}",
            showarrow=False,
            yshift=15,
            font=dict(size=9, color="#666688"),
        )

    fig.update_layout(**layout)
    return fig
