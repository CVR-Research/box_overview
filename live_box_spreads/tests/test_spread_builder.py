from datetime import datetime, timedelta, timezone

from live_box_spreads.config import Config
from live_box_spreads.core.types import OptionQuote
from live_box_spreads.engine.spread_builder import SpreadBuilder


def test_build_from_quotes_generates_rows():
    config = Config.from_dict({
        "tickers": ["SPX"],
        "expiries_per_ticker": 1,
        "min_volume": 0,
        "min_open_interest": 0,
        "min_strike_gap": 1.0,
        "max_strike_gap": 100.0,
    })
    builder = SpreadBuilder(config)
    now = datetime.now(timezone.utc)
    expiry = (now + timedelta(days=180)).date().isoformat()
    quotes = [
        OptionQuote("SPX", expiry, 4000.0, "call", 10.0, 11.0, 10.5, 10.5, 100, 100),
        OptionQuote("SPX", expiry, 4000.0, "put", 9.5, 10.5, 10.0, 10.0, 100, 100),
        OptionQuote("SPX", expiry, 4005.0, "call", 8.0, 9.0, 8.5, 8.5, 100, 100),
        OptionQuote("SPX", expiry, 4005.0, "put", 11.0, 12.0, 11.5, 11.5, 100, 100),
    ]
    df = builder.build_from_quotes("SPX", expiry, 4000.0, quotes, now)
    assert not df.is_empty()
    assert df.height > 0
    assert "mid_rate" in df.columns
    assert "lend_rate" in df.columns
    assert "borrow_rate" in df.columns
    assert "moneyness_mid" in df.columns
    # Lend rate positive, borrow rate negative
    import math
    row = df.row(0, named=True)
    if not math.isnan(row["lend_rate"]):
        assert row["lend_rate"] > 0
    if not math.isnan(row["borrow_rate"]):
        assert row["borrow_rate"] < 0


def test_build_from_quotes_empty():
    config = Config.from_dict({"min_strike_gap": 1.0})
    builder = SpreadBuilder(config)
    df = builder.build_from_quotes("SPX", "2026-01-18", 4000.0, [], datetime.now(timezone.utc))
    assert df.is_empty()


def test_build_respects_strike_gap():
    config = Config.from_dict({
        "min_strike_gap": 100.0,
        "max_strike_gap": 200.0,
        "min_volume": 0,
        "min_open_interest": 0,
    })
    builder = SpreadBuilder(config)
    now = datetime.now(timezone.utc)
    expiry = (now + timedelta(days=180)).date().isoformat()
    quotes = [
        OptionQuote("SPX", expiry, 4000.0, "call", 10.0, 11.0, 10.5, 10.5, 100, 100),
        OptionQuote("SPX", expiry, 4000.0, "put", 9.5, 10.5, 10.0, 10.0, 100, 100),
        # Width = 5, which is < min_strike_gap of 100
        OptionQuote("SPX", expiry, 4005.0, "call", 8.0, 9.0, 8.5, 8.5, 100, 100),
        OptionQuote("SPX", expiry, 4005.0, "put", 11.0, 12.0, 11.5, 11.5, 100, 100),
    ]
    df = builder.build_from_quotes("SPX", expiry, 4000.0, quotes, now)
    assert df.is_empty()
