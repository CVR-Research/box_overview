import polars as pl
import plotly.graph_objects as go

from live_box_spreads.dashboard.charts.surfaces import make_rate_curve, make_rate_heatmap
from live_box_spreads.dashboard.charts.term_structure import make_term_structure
from live_box_spreads.dashboard.charts.skew import make_lend_borrow_chart
from live_box_spreads.dashboard.table import make_top_table


def _sample_df():
    return pl.DataFrame({
        "ticker": ["SPX"] * 4,
        "expiry": ["2026-01-01"] * 4,
        "snapshot_time": [
            "2026-01-01T10:00:00",
            "2026-01-01T10:00:00",
            "2026-01-01T10:01:00",
            "2026-01-01T10:01:00",
        ],
        "mid_strike": [4000.0, 4050.0, 4000.0, 4050.0],
        "mid_rate": [0.02, 0.025, 0.021, 0.026],
        "lend_rate": [0.019, 0.024, 0.020, 0.025],
        "borrow_rate": [-0.021, -0.026, -0.022, -0.027],
        "total_leg_volume": [100, 150, 110, 160],
        "moneyness_mid": [1.0, 1.01, 1.0, 1.01],
        "width": [50.0, 50.0, 50.0, 50.0],
        "kl": [4000.0, 4025.0, 4000.0, 4025.0],
        "ks": [4050.0, 4075.0, 4050.0, 4075.0],
        "min_leg_volume": [100, 100, 100, 100],
    })


def test_make_rate_curve():
    fig = make_rate_curve(_sample_df(), "2026-01-01")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0


def test_make_rate_curve_empty():
    fig = make_rate_curve(pl.DataFrame())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_make_rate_heatmap():
    fig = make_rate_heatmap(_sample_df(), "2026-01-01")
    assert isinstance(fig, go.Figure)


def test_make_term_structure():
    fig = make_term_structure(_sample_df())
    assert isinstance(fig, go.Figure)


def test_make_term_structure_empty():
    fig = make_term_structure(pl.DataFrame())
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 0


def test_make_lend_borrow_chart():
    fig = make_lend_borrow_chart(_sample_df(), expiry="2026-01-01")
    assert isinstance(fig, go.Figure)


def test_make_top_table():
    df = pl.DataFrame({
        "ticker": ["SPX", "SPX"],
        "expiry": ["2026-01-01", "2026-01-01"],
        "kl": [4000.0, 4010.0],
        "ks": [4050.0, 4060.0],
        "width": [50.0, 50.0],
        "mid_rate": [0.045, 0.042],
        "lend_rate": [0.040, 0.038],
        "borrow_rate": [-0.050, -0.046],
        "min_leg_volume": [100, 100],
    })
    records = make_top_table(df)
    assert records
    assert "%" in str(records[0]["mid_rate"])


def test_make_top_table_empty():
    assert make_top_table(pl.DataFrame()) == []
