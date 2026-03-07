"""Dash app factory. No global state."""
from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import Dash

from live_box_spreads.config import Config
from live_box_spreads.dashboard.callbacks import register_callbacks
from live_box_spreads.dashboard.layout import build_layout
from live_box_spreads.sources.protocol import DataSource


def create_app(
    source: DataSource,
    config: Config,
    *,
    update_interval_ms: int = 60000,
) -> Dash:
    """Create and configure the Dash application."""
    app = Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        suppress_callback_exceptions=True,
    )
    app.title = "Live Box Spreads"
    app.layout = build_layout(config, update_interval_ms=update_interval_ms)
    register_callbacks(app, source, config)
    return app
