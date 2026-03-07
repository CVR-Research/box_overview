import polars as pl

from live_box_spreads.config import Config
from live_box_spreads.dashboard.app import create_app


class MockSource:
    """Minimal DataSource for testing."""

    def __init__(self, df: pl.DataFrame | None = None):
        self._df = df if df is not None else pl.DataFrame()

    def latest_spreads(self) -> pl.DataFrame:
        return self._df

    def status(self) -> dict:
        return {"source": "mock", "rows": self._df.height}


def test_create_app_returns_dash_instance():
    config = Config.from_dict({"tickers": ["SPX"]})
    source = MockSource()
    app = create_app(source, config)
    assert app.title == "Live Box Spreads"
    assert app.layout is not None


def test_create_app_with_data():
    df = pl.DataFrame({
        "ticker": ["SPX"],
        "expiry": ["2026-01-01"],
        "snapshot_time": ["2026-01-01T10:00:00"],
        "mid_strike": [4000.0],
        "mid_rate": [0.02],
        "bid_rate": [0.019],
        "ask_rate": [0.021],
        "total_leg_volume": [100],
        "moneyness_mid": [1.0],
        "width": [50.0],
        "kl": [4000.0],
        "ks": [4050.0],
        "min_leg_volume": [100],
    })
    config = Config.from_dict({"tickers": ["SPX"]})
    source = MockSource(df)
    app = create_app(source, config)
    assert app.title == "Live Box Spreads"
