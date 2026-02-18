from datetime import datetime, timedelta, timezone

import msgpack

from alpaca_client import OptionQuote
from alpaca_stream import StreamQuoteCache, decode_stream_payload, quote_from_stream_message
from dashboard import app as dashboard_app
from ingest import Config as IngestConfig


def test_decode_msgpack_payload_and_quote_parse():
    payload = [
        {
            "T": "q",
            "S": "SPX260118C04000000",
            "bp": 1.0,
            "ap": 1.2,
            "p": 1.1,
            "v": 10,
        }
    ]
    raw = msgpack.packb(payload)
    decoded = decode_stream_payload(raw, use_msgpack=True)
    assert decoded
    quote = quote_from_stream_message(decoded[0])
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


def test_build_spreads_from_quotes_generates_rows():
    config = IngestConfig.from_dict(
        {
            "tickers": ["SPX"],
            "expiries_per_ticker": 1,
            "min_volume": 0,
            "min_open_interest": 0,
            "min_strike_gap": 1.0,
            "max_strike_gap": 10.0,
        }
    )
    now = datetime.now(timezone.utc)
    expiry = (now + timedelta(days=180)).date().isoformat()
    quotes = [
        OptionQuote("SPX", expiry, 4000.0, "call", 10.0, 11.0, 10.5, 10.5, 100, 100),
        OptionQuote("SPX", expiry, 4000.0, "put", 9.5, 10.5, 10.0, 10.0, 100, 100),
        OptionQuote("SPX", expiry, 4005.0, "call", 8.0, 9.0, 8.5, 8.5, 100, 100),
        OptionQuote("SPX", expiry, 4005.0, "put", 11.0, 12.0, 11.5, 11.5, 100, 100),
    ]
    df = dashboard_app.build_spreads_from_quotes("SPX", expiry, 4000.0, quotes, config, now)
    assert not df.is_empty()
    assert df.height > 0
