from alpaca.data.enums import OptionsFeed

from live_box_spreads.core.types import OptionQuote
from live_box_spreads.providers.alpaca_stream import (
    StreamQuoteCache,
    _resolve_options_feed,
    quote_from_stream_message,
)


def test_resolve_options_feed():
    assert _resolve_options_feed("opra") == OptionsFeed.OPRA
    assert _resolve_options_feed("OPRA") == OptionsFeed.OPRA
    assert _resolve_options_feed("indicative") == OptionsFeed.INDICATIVE
    assert _resolve_options_feed("INDICATIVE") == OptionsFeed.INDICATIVE


def test_quote_parse_from_stream_message():
    msg = {
        "T": "q",
        "S": "SPX260118C04000000",
        "bp": 1.0,
        "ap": 1.2,
        "p": 1.1,
        "v": 10,
    }
    quote = quote_from_stream_message(msg)
    assert quote is not None
    assert quote.ticker == "SPX"
    assert quote.option_type == "call"
    assert quote.strike == 4000.0


def test_stream_cache_version_increments():
    cache = StreamQuoteCache()
    q = OptionQuote(
        ticker="SPX",
        expiry="2026-01-18",
        strike=4000.0,
        option_type="call",
        bid=1.0,
        ask=1.2,
        last=1.1,
        mark=1.1,
        volume=10,
        open_interest=100,
    )
    v0 = cache.version()
    cache.update(q)
    assert cache.version() == v0 + 1


def test_stream_cache_snapshot():
    cache = StreamQuoteCache()
    q1 = OptionQuote("SPX", "2026-01-18", 4000.0, "call", 1.0, 1.2, 1.1, 1.1, 10, 100)
    q2 = OptionQuote("SPX", "2026-01-18", 4000.0, "put", 2.0, 2.2, 2.1, 2.1, 20, 200)
    q3 = OptionQuote("SPX", "2026-02-15", 4000.0, "call", 3.0, 3.2, 3.1, 3.1, 30, 300)
    cache.update(q1)
    cache.update(q2)
    cache.update(q3)
    result = cache.snapshot("SPX", "2026-01-18")
    assert len(result) == 2
