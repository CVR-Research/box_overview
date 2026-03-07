"""Alpaca WebSocket stream for real-time option quotes."""
from __future__ import annotations

import logging
import os
import threading
from typing import Any, Dict, Iterable, List, Optional

from alpaca.data.enums import OptionsFeed
from alpaca.data.live import OptionDataStream

from live_box_spreads.core.opra import extract_underlying, parse_opra_symbol
from live_box_spreads.core.types import OptionQuote

LOGGER = logging.getLogger(__name__)

DEFAULT_FEED = os.environ.get("ALPACA_DATA_FEED", "indicative")
DEFAULT_STREAM_URL = os.environ.get("ALPACA_STREAM_URL")


class StreamQuoteKey:
    """Hashable key for the quote cache."""

    __slots__ = ("ticker", "expiry", "strike", "option_type", "_hash")

    def __init__(self, ticker: str, expiry: str, strike: float, option_type: str) -> None:
        self.ticker = ticker
        self.expiry = expiry
        self.strike = strike
        self.option_type = option_type
        self._hash = hash((ticker, expiry, strike, option_type))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StreamQuoteKey):
            return NotImplemented
        return (
            self.ticker == other.ticker
            and self.expiry == other.expiry
            and self.strike == other.strike
            and self.option_type == other.option_type
        )

    def __hash__(self) -> int:
        return self._hash


class StreamQuoteCache:
    """Thread-safe quote cache keyed by (ticker, expiry, strike, option_type)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._quotes: Dict[StreamQuoteKey, OptionQuote] = {}
        self._version = 0

    def update(self, quote: OptionQuote) -> None:
        key = StreamQuoteKey(quote.ticker, quote.expiry, float(quote.strike), quote.option_type)
        with self._lock:
            existing = self._quotes.get(key)
            if existing is None:
                self._quotes[key] = quote
            else:
                self._quotes[key] = _merge_quotes(existing, quote)
            self._version += 1

    def snapshot(self, ticker: str, expiry: str) -> List[OptionQuote]:
        with self._lock:
            return [
                q for k, q in self._quotes.items()
                if k.ticker == ticker and k.expiry == expiry
            ]

    def version(self) -> int:
        with self._lock:
            return self._version


class AlpacaStreamRunner:
    """Run Alpaca option data stream in a background thread."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: Iterable[str],
        cache: StreamQuoteCache,
        *,
        feed: str = DEFAULT_FEED,
        stream_url: Optional[str] = DEFAULT_STREAM_URL,
        websocket_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = list(symbols)
        self.cache = cache
        self.feed = feed
        self.stream_url = stream_url
        self.websocket_params = websocket_params
        self._thread: Optional[threading.Thread] = None
        self._stream: Optional[OptionDataStream] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()

    def _run(self) -> None:
        feed = _resolve_options_feed(self.feed)
        self._stream = OptionDataStream(
            self.api_key,
            self.api_secret,
            raw_data=True,
            feed=feed,
            websocket_params=self.websocket_params,
            url_override=self.stream_url,
        )

        async def _on_quote(data: Any) -> None:
            msg = _coerce_stream_message(data)
            if not msg:
                return
            quote = quote_from_stream_message(msg)
            if quote is not None:
                self.cache.update(quote)

        if not self.symbols:
            LOGGER.warning("No option symbols provided for stream subscription")
        else:
            self._stream.subscribe_quotes(_on_quote, *self.symbols)
        self._stream.run()


# ------------------------------------------------------------------
# Parsing helpers
# ------------------------------------------------------------------

def _resolve_options_feed(feed: str) -> OptionsFeed:
    normalized = (feed or "").strip().lower()
    if normalized == "opra":
        return OptionsFeed.OPRA
    if normalized == "indicative":
        return OptionsFeed.INDICATIVE
    LOGGER.warning("Unknown options feed %r; defaulting to INDICATIVE", feed)
    return OptionsFeed.INDICATIVE


def _coerce_stream_message(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data
    symbol = getattr(data, "symbol", None)
    if not symbol:
        return {}
    timestamp = getattr(data, "timestamp", None)
    return {
        "S": symbol,
        "bp": getattr(data, "bid_price", None),
        "ap": getattr(data, "ask_price", None),
        "bs": getattr(data, "bid_size", None),
        "as": getattr(data, "ask_size", None),
        "t": timestamp.isoformat() if timestamp else None,
    }


def quote_from_stream_message(msg: Dict[str, Any]) -> Optional[OptionQuote]:
    """Parse a WebSocket stream message into an OptionQuote."""
    symbol = _coerce_str(msg, ["symbol", "S", "sym"])
    if not symbol:
        return None
    expiry = _coerce_str(msg, ["expiration_date", "expiry", "exp", "exp_date"])
    strike_val = _coerce_float(msg, ["strike", "strike_price", "k"])
    option_type = _coerce_str(msg, ["option_type", "right", "side", "cp", "put_call"])
    if not (expiry and strike_val and option_type):
        parsed = parse_opra_symbol(symbol)
        if parsed:
            expiry = expiry or parsed[1]
            strike_val = strike_val or parsed[2]
            option_type = option_type or parsed[3]
    if not (expiry and strike_val and option_type):
        return None
    option_type = "call" if option_type.lower().startswith("c") else "put"
    bid = _coerce_float(msg, ["bp", "bid", "bid_price"], default=0.0) or 0.0
    ask = _coerce_float(msg, ["ap", "ask", "ask_price"], default=0.0) or 0.0
    last = _coerce_float(msg, ["p", "last", "last_price"], default=0.0) or 0.0
    mark = _coerce_float(msg, ["mid", "mark"], default=0.0) or 0.0
    volume = int(_coerce_float(msg, ["v", "volume"], default=0.0) or 0.0)
    open_interest = int(_coerce_float(msg, ["oi", "open_interest"], default=0.0) or 0.0)
    bid_size = _coerce_float(msg, ["bs", "bid_size"], default=None)
    ask_size = _coerce_float(msg, ["as", "ask_size"], default=None)
    return OptionQuote(
        ticker=extract_underlying(symbol),
        expiry=expiry,
        strike=float(strike_val),
        option_type=option_type,
        bid=bid,
        ask=ask,
        last=last,
        mark=mark,
        volume=volume,
        open_interest=open_interest,
        bid_size=int(bid_size) if bid_size is not None else None,
        ask_size=int(ask_size) if ask_size is not None else None,
        symbol=symbol,
    )


def _merge_quotes(existing: OptionQuote, incoming: OptionQuote) -> OptionQuote:
    return OptionQuote(
        ticker=existing.ticker,
        expiry=existing.expiry,
        strike=existing.strike,
        option_type=existing.option_type,
        bid=incoming.bid or existing.bid,
        ask=incoming.ask or existing.ask,
        last=incoming.last or existing.last,
        mark=incoming.mark or existing.mark,
        volume=incoming.volume or existing.volume,
        open_interest=incoming.open_interest or existing.open_interest,
        bid_size=incoming.bid_size if incoming.bid_size is not None else existing.bid_size,
        ask_size=incoming.ask_size if incoming.ask_size is not None else existing.ask_size,
        symbol=incoming.symbol or existing.symbol,
    )


def _coerce_float(
    raw: Dict[str, Any], keys: Iterable[str], default: Optional[float] = None
) -> Optional[float]:
    for key in keys:
        if key not in raw:
            continue
        val = raw[key]
        if val in (None, ""):
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return default


def _coerce_str(raw: Dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        val = raw.get(key)
        if isinstance(val, str) and val:
            return val
    return ""
