"""Dashboard layout construction. Pure function, no side effects."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

from live_box_spreads.config import Config

# Shared dark style for chart cards
CARD_STYLE = {"backgroundColor": "#1a1a2e", "border": "1px solid #2d2d44"}
GRAPH_CONFIG = {"displayModeBar": True, "displaylogo": False}

# Force dark styling on dcc.Dropdown, inputs, and range sliders
def build_layout(config: Config, update_interval_ms: int = 60000) -> dbc.Container:
    """Return the complete Dash layout with dark theme."""
    return dbc.Container(
        [
            # Hidden stores and interval
            dcc.Store(id="snapshot-store"),
            dcc.Store(id="status-store"),
            dcc.Interval(id="refresh-interval", interval=update_interval_ms, n_intervals=0),

            # Navbar
            dbc.Navbar(
                dbc.Container(
                    [
                        dbc.NavbarBrand(
                            "Box Spread Monitor",
                            className="fs-4 fw-bold",
                            style={"letterSpacing": "0.5px"},
                        ),
                        html.Div(
                            id="stream-status",
                            className="text-muted small",
                            style={"maxWidth": "600px", "textAlign": "right"},
                        ),
                    ],
                    fluid=True,
                    className="d-flex justify-content-between align-items-center",
                ),
                color="#0f0f1a",
                dark=True,
                className="mb-3",
                style={"borderBottom": "1px solid #2d2d44"},
            ),

            # Filter bar
            dbc.Card(
                dbc.CardBody(
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    html.Label("Ticker", className="text-muted small mb-1"),
                                    dcc.Dropdown(
                                        id="ticker-selector",
                                        options=[{"label": t, "value": t} for t in config.tickers],
                                        value=config.tickers[0] if config.tickers else None,
                                        clearable=False,
                                        className="dash-dropdown-dark",
                                    ),
                                ],
                                md=2,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Expiry", className="text-muted small mb-1"),
                                    dcc.Dropdown(id="expiry-dropdown", clearable=False,
                                                 className="dash-dropdown-dark"),
                                ],
                                md=2,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Min Leg Volume", className="text-muted small mb-1"),
                                    dbc.Input(
                                        id="min-volume-input",
                                        type="number",
                                        value=config.min_volume,
                                        min=0,
                                        step=1,
                                        size="sm",
                                    ),
                                ],
                                md=2,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Width Range", className="text-muted small mb-1"),
                                    dcc.RangeSlider(
                                        id="width-range",
                                        min=0,
                                        max=400,
                                        step=5,
                                        value=[config.min_strike_gap, config.max_strike_gap],
                                        marks={0: "0", 50: "50", 100: "100", 200: "200", 400: "400"},
                                        tooltip={"placement": "bottom"},
                                    ),
                                ],
                                md=3,
                            ),
                            dbc.Col(
                                [
                                    html.Label("History (min)", className="text-muted small mb-1"),
                                    dbc.Input(
                                        id="history-slider",
                                        type="number",
                                        value=config.surface_history_minutes,
                                        min=5,
                                        max=1440,
                                        step=5,
                                        size="sm",
                                    ),
                                ],
                                md=1,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Moneyness Band (±%)", className="text-muted small mb-1"),
                                    dcc.Slider(
                                        id="moneyness-band",
                                        min=5,
                                        max=100,
                                        step=5,
                                        value=30,
                                        marks={5: "5%", 25: "25%", 50: "50%", 100: "100%"},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                    ),
                                ],
                                md=2,
                            ),
                            dbc.Col(
                                [
                                    html.Label("Skew Expiry", className="text-muted small mb-1"),
                                    dcc.Dropdown(id="skew-expiry-dropdown", clearable=False,
                                                 className="dash-dropdown-dark"),
                                ],
                                md=2,
                            ),
                        ],
                        className="g-3 align-items-end",
                    ),
                    className="py-2 px-3",
                ),
                className="mb-3",
                style=CARD_STYLE,
            ),

            # Row 1: Rate Curve + Term Structure
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    "Implied Rate vs Strike",
                                    className="py-2 small fw-bold",
                                    style={"backgroundColor": "#12122a", "borderBottom": "1px solid #2d2d44"},
                                ),
                                dbc.CardBody(
                                    dcc.Graph(id="rate-curve", config=GRAPH_CONFIG,
                                              style={"height": "380px"}),
                                    className="p-1",
                                ),
                            ],
                            style=CARD_STYLE,
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    "Rate Term Structure",
                                    className="py-2 small fw-bold",
                                    style={"backgroundColor": "#12122a", "borderBottom": "1px solid #2d2d44"},
                                ),
                                dbc.CardBody(
                                    dcc.Graph(id="term-structure", config=GRAPH_CONFIG,
                                              style={"height": "380px"}),
                                    className="p-1",
                                ),
                            ],
                            style=CARD_STYLE,
                        ),
                        md=6,
                    ),
                ],
                className="mb-3",
            ),

            # Row 2: Rate Heatmap + Bid/Ask Spread
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    "Rate Heatmap (Strike Pairs)",
                                    className="py-2 small fw-bold",
                                    style={"backgroundColor": "#12122a", "borderBottom": "1px solid #2d2d44"},
                                ),
                                dbc.CardBody(
                                    dcc.Graph(id="rate-heatmap", config=GRAPH_CONFIG,
                                              style={"height": "380px"}),
                                    className="p-1",
                                ),
                            ],
                            style=CARD_STYLE,
                        ),
                        md=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    "Lend vs Borrow Rate",
                                    className="py-2 small fw-bold",
                                    style={"backgroundColor": "#12122a", "borderBottom": "1px solid #2d2d44"},
                                ),
                                dbc.CardBody(
                                    dcc.Graph(id="skew-chart", config=GRAPH_CONFIG,
                                              style={"height": "380px"}),
                                    className="p-1",
                                ),
                            ],
                            style=CARD_STYLE,
                        ),
                        md=6,
                    ),
                ],
                className="mb-3",
            ),

            # Row 3: Top Opportunities Table
            dbc.Row(
                dbc.Col(
                    dbc.Card(
                        [
                            dbc.CardHeader(
                                "Top Opportunities",
                                className="py-2 small fw-bold",
                                style={"backgroundColor": "#12122a", "borderBottom": "1px solid #2d2d44"},
                            ),
                            dbc.CardBody(
                                dash_table.DataTable(
                                    id="opportunities-table",
                                    columns=[
                                        {"name": "Ticker", "id": "ticker"},
                                        {"name": "Expiry", "id": "expiry"},
                                        {"name": "Lower K", "id": "kl", "type": "numeric", "format": {"specifier": ",.0f"}},
                                        {"name": "Upper K", "id": "ks", "type": "numeric", "format": {"specifier": ",.0f"}},
                                        {"name": "Width", "id": "width", "type": "numeric", "format": {"specifier": ",.0f"}},
                                        {"name": "Mid Rate", "id": "mid_rate"},
                                        {"name": "Lend Rate", "id": "lend_rate"},
                                        {"name": "Borrow Rate", "id": "borrow_rate"},
                                        {"name": "Min Leg Vol", "id": "min_leg_volume", "type": "numeric"},
                                    ],
                                    style_table={"overflowX": "auto"},
                                    style_header={
                                        "backgroundColor": "#12122a",
                                        "color": "#8888aa",
                                        "fontWeight": "600",
                                        "fontSize": "12px",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.5px",
                                        "border": "1px solid #2d2d44",
                                        "padding": "8px 12px",
                                    },
                                    style_cell={
                                        "backgroundColor": "#1a1a2e",
                                        "color": "#e0e0e0",
                                        "border": "1px solid #2d2d44",
                                        "fontSize": "13px",
                                        "fontFamily": "'JetBrains Mono', 'Fira Code', monospace",
                                        "padding": "6px 12px",
                                        "textAlign": "right",
                                    },
                                    style_cell_conditional=[
                                        {"if": {"column_id": "ticker"}, "textAlign": "left"},
                                        {"if": {"column_id": "expiry"}, "textAlign": "left"},
                                    ],
                                    page_size=15,
                                ),
                                className="p-2",
                            ),
                        ],
                        style=CARD_STYLE,
                    ),
                    md=12,
                ),
                className="mb-3",
            ),
        ],
        fluid=True,
        style={"backgroundColor": "#0f0f1a", "minHeight": "100vh", "padding": "0"},
    )
