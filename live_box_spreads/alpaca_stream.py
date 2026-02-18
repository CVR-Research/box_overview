"""Alpaca websocket stream helpers for real-time option quotes."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import date
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Tuple

import msgpack
import websockets

from alpaca_client import OptionQuote

LOGGER = logging.getLogger(__name__)

DEFAULT_STREAM_URL = os.environ.get(
    "ALPACA_STREAM_URL",
    "wss://stream.data.sandbox.alpaca.markets/v1beta1/indicative",
)

OPRA_PATTERN = re.compile(r"^(?P<root>[A-Z0-9]{1,6})(?P<date>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")


@dataclass(frozen=True)
class StreamQuoteKey:
    ticker: str
    expiry: str
    strike: float
    option_type: str


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
                self._version += 1
                return
            merged = _merge_quotes(existing, quote)
            self._quotes[key] = merged
            self._version += 1

    def snapshot(self, ticker: str, expiry: str) -> List[OptionQuote]:
        with self._lock:
            return [
                q
                for k, q in self._quotes.items()
                if k.ticker == ticker and k.expiry == expiry
            ]

    def version(self) -> int:
        with self._lock:
            return self._version


class AlpacaStreamClient:
    """Minimal websocket client for Alpaca market data streams."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        stream_url: str = DEFAULT_STREAM_URL,
        ping_interval: float = 20.0,
        ping_timeout: float = 20.0,
        use_msgpack: bool = True,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.stream_url = stream_url
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.use_msgpack = use_msgpack
        self._ws: Optional[websockets.WebSocketClientProtocol] = None

    async def connect(self) -> None:
        headers = None
        if self.use_msgpack:
            headers = {
                "Content-Type": "application/msgpack",
                "Accept": "application/msgpack",
            }
        self._ws = await websockets.connect(
            self.stream_url,
            ping_interval=self.ping_interval,
            ping_timeout=self.ping_timeout,
            extra_headers=headers,
        )
        await self._authenticate()

    async def _authenticate(self) -> None:
        assert self._ws is not None
        payload = {"action": "auth", "key": self.api_key, "secret": self.api_secret}
        await self._send(payload)
        await self._await_auth_reply()

    async def _await_auth_reply(self) -> None:
        assert self._ws is not None
        for _ in range(3):
            raw = await self._ws.recv()
            for msg in _decode_messages(raw):
                status = msg.get("status") or msg.get("msg") or msg.get("message")
                if status and "auth" in str(status).lower() and "success" in str(status).lower():
                    return
                if msg.get("T") == "success" and "authenticated" in str(msg.get("msg", "")).lower():
                    return
                if msg.get("T") == "error":
                    raise RuntimeError(f"Stream auth failed: {msg}")
        LOGGER.warning("Stream auth response not explicit; continuing")

    async def subscribe_quotes(self, symbols: Iterable[str]) -> None:
        assert self._ws is not None
        symbols_list = [s for s in symbols if s]
        if not symbols_list:
            LOGGER.warning("No option symbols provided for stream subscription")
            return
        payload = {"action": "subscribe", "quotes": symbols_list}
        await self._send(payload)

    async def listen(self) -> AsyncIterator[Dict[str, Any]]:
        assert self._ws is not None
        async for raw in self._ws:
            for msg in decode_stream_payload(raw, use_msgpack=self.use_msgpack):
                yield msg

    async def _send(self, payload: Dict[str, Any]) -> None:
        assert self._ws is not None
        if self.use_msgpack:
            await self._ws.send(msgpack.packb(payload))
        else:
            await self._ws.send(json.dumps(payload))


class AlpacaStreamRunner:
    """Run websocket stream in a background thread and update a cache."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbols: Iterable[str],
        cache: StreamQuoteCache,
        *,
        stream_url: str = DEFAULT_STREAM_URL,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = list(symbols)
        self.cache = cache
        self.stream_url = stream_url
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        client = AlpacaStreamClient(
            self.api_key,
            self.api_secret,
            stream_url=self.stream_url,
        )
        await client.connect()
        await client.subscribe_quotes(self.symbols)
        async for msg in client.listen():
            quote = quote_from_stream_message(msg)
            if quote is not None:
                self.cache.update(quote)


# ----------------------------------------------------------------------
# Parsing helpers
# ----------------------------------------------------------------------

def quote_from_stream_message(msg: Dict[str, Any]) -> Optional[OptionQuote]:
    symbol = _coerce_str(msg, ["symbol", "S", "sym"])
    if not symbol:
        return None
    expiry = _coerce_str(msg, ["expiration_date", "expiry", "exp", "exp_date"])
    strike_val = _coerce_float(msg, ["strike", "strike_price", "k"])
    option_type = _coerce_str(msg, ["option_type", "right", "side", "cp", "put_call"])
    if not (expiry and strike_val and option_type):
        parsed = _parse_opra_symbol(symbol)
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
        ticker=_parse_underlying(symbol),
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


def decode_stream_payload(raw: Any, *, use_msgpack: bool = True) -> List[Dict[str, Any]]:
    if use_msgpack and isinstance(raw, (bytes, bytearray)):
        try:
            payload = msgpack.unpackb(raw, raw=False)
        except Exception:
            return []
    else:
        try:
            payload = json.loads(raw)
        except Exception:
            return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        return [payload]
    return []


def _parse_opra_symbol(symbol: str) -> Optional[Tuple[str, str, float, str]]:
    cleaned = symbol.replace(" ", "").upper()
    match = OPRA_PATTERN.match(cleaned)
    if not match:
        return None
    root = match.group("root").strip()
    yymmdd = match.group("date")
    year = 2000 + int(yymmdd[0:2])
    month = int(yymmdd[2:4])
    day = int(yymmdd[4:6])
    try:
        expiry = date(year, month, day).isoformat()
    except ValueError:
        return None
    option_type = "call" if match.group("cp") == "C" else "put"
    strike = int(match.group("strike")) / 1000.0
    return root, expiry, strike, option_type


def _parse_underlying(symbol: str) -> str:
    parsed = _parse_opra_symbol(symbol)
    if parsed:
        return parsed[0]
    return symbol


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


def _coerce_float(raw: Dict[str, Any], keys: Iterable[str], default: Optional[float] = None) -> Optional[float]:
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


__all__ = [
    "AlpacaStreamClient",
    "AlpacaStreamRunner",
    "StreamQuoteCache",
    "quote_from_stream_message",
    "decode_stream_payload",
]
