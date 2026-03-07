from live_box_spreads.core.types import OptionQuote
from live_box_spreads.providers.alpaca_stream import StreamQuoteCache


def test_cache_update_and_snapshot():
    cache = StreamQuoteCache()
    q1 = OptionQuote("SPX", "2026-01-18", 4000.0, "call", 1.0, 1.2, 1.1, 1.1, 10, 100)
    q2 = OptionQuote("SPX", "2026-01-18", 4050.0, "put", 2.0, 2.2, 2.1, 2.1, 20, 200)
    cache.update(q1)
    cache.update(q2)
    result = cache.snapshot("SPX", "2026-01-18")
    assert len(result) == 2
    assert cache.version() == 2


def test_cache_merge_on_update():
    cache = StreamQuoteCache()
    q1 = OptionQuote("SPX", "2026-01-18", 4000.0, "call", 1.0, 1.2, 1.1, 1.1, 10, 100)
    cache.update(q1)
    # Update with new bid/ask
    q2 = OptionQuote("SPX", "2026-01-18", 4000.0, "call", 1.5, 1.8, 0.0, 0.0, 0, 0)
    cache.update(q2)
    result = cache.snapshot("SPX", "2026-01-18")
    assert len(result) == 1
    # Bid/ask should be updated, last/mark should fall back to original
    assert result[0].bid == 1.5
    assert result[0].ask == 1.8
    assert result[0].last == 1.1  # from original (incoming was 0)
    assert result[0].mark == 1.1  # from original (incoming was 0)
