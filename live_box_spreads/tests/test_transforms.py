import polars as pl

from live_box_spreads.core.types import OptionQuote
from live_box_spreads.engine.transforms import derive_spot_from_chain, pivot_calls_puts, quotes_to_frame


def _make_quotes():
    return [
        OptionQuote("SPX", "2026-01-18", 4000.0, "call", 10.0, 12.0, 11.0, 11.0, 100, 500),
        OptionQuote("SPX", "2026-01-18", 4000.0, "put", 8.0, 10.0, 9.0, 9.0, 80, 400),
        OptionQuote("SPX", "2026-01-18", 4050.0, "call", 5.0, 7.0, 6.0, 6.0, 60, 300),
        OptionQuote("SPX", "2026-01-18", 4050.0, "put", 12.0, 14.0, 13.0, 13.0, 70, 350),
    ]


def test_quotes_to_frame():
    df = quotes_to_frame(_make_quotes())
    assert df.height == 4
    assert "mid" in df.columns
    # Mid should be (bid + ask) / 2 for valid bid/ask
    call_row = df.filter((pl.col("strike") == 4000.0) & (pl.col("type") == "call"))
    assert call_row["mid"][0] == 11.0  # (10 + 12) / 2


def test_quotes_to_frame_empty():
    df = quotes_to_frame([])
    assert df.is_empty()


def test_pivot_calls_puts():
    df = quotes_to_frame(_make_quotes())
    merged = pivot_calls_puts(df)
    assert merged.height == 2
    assert "call_mid" in merged.columns
    assert "put_mid" in merged.columns
    # Strikes should be 4000 and 4050
    strikes = sorted(merged["strike"].to_list())
    assert strikes == [4000.0, 4050.0]


def test_pivot_empty():
    result = pivot_calls_puts(pl.DataFrame())
    assert result.is_empty()


def test_derive_spot_from_chain():
    quotes = _make_quotes()
    spot = derive_spot_from_chain(quotes)
    assert spot is not None
    # Forward = K + C - P at the most ATM strike
    # At K=4000: C=11.0, P=9.0 -> F=4002
    # At K=4050: C=6.0, P=13.0 -> F=4043
    # |C-P| at K=4000 is 2.0, at K=4050 is 7.0, so K=4000 is picked
    assert abs(spot - 4002.0) < 0.01


def test_derive_spot_empty():
    assert derive_spot_from_chain([]) is None
