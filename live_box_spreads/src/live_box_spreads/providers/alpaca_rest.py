"""Alpaca Market Data REST API provider.

Implements the DataProvider protocol. Keeps a session alive, retries
transient failures, and exposes the small API the pipeline needs.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

from live_box_spreads.core.opra import parse_opra_symbol
from live_box_spreads.core.types import OptionQuote

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = os.environ.get("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets")
DEFAULT_FEED = os.environ.get("ALPACA_DATA_FEED", "indicative")
DEFAULT_MIN_REQUEST_INTERVAL = float(
    os.environ.get("ALPACA_MIN_REQUEST_INTERVAL_SECONDS", "2.0")
)


class AlpacaRestProvider:
    """Authenticated wrapper over Alpaca market data endpoints."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        feed: str = DEFAULT_FEED,
        timeout: float = 15.0,
        max_retries: int = 3,
        min_request_interval_seconds: float = DEFAULT_MIN_REQUEST_INTERVAL,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.feed = feed
        self.timeout = timeout
        self.max_retries = max_retries
        self.min_request_interval_seconds = max(0.0, float(min_request_interval_seconds))
        self._rate_limit_lock = threading.Lock()
        self._last_request_ts = 0.0
        self._session = requests.Session()

    # --- DataProvider protocol ---

    def get_option_chain(
        self, symbol: str, *, page_limit: int = 100, max_pages: int = 10
    ) -> List[dict]:
        all_items: List[dict] = []
        next_page_token: Optional[str] = None
        for _ in range(max_pages):
            params: Dict[str, Any] = {"limit": page_limit, "feed": self.feed}
            if next_page_token:
                params["page_token"] = next_page_token
            data = self._request(
                "GET",
                f"/v1beta1/options/snapshots/{symbol}",
                params=params,
            )
            all_items.extend(_extract_items(data))
            next_page_token = data.get("next_page_token") if isinstance(data, dict) else None
            if not next_page_token:
                break
        if not all_items:
            LOGGER.warning("Option chain empty for %s", symbol)
        return all_items

    def get_underlying_price(self, symbol: str) -> Optional[float]:
        paths = (
            (f"/v2/stocks/{symbol}/trades/latest", {}),
            (f"/v2/stocks/{symbol}/quotes/latest", {}),
            (f"/v2/stocks/{symbol}/snapshot", {}),
            (f"/v1beta1/options/snapshots/{symbol}", {"feed": self.feed}),
        )
        for path, params in paths:
            try:
                data = self._request("GET", path, params=params)
            except Exception:
                continue
            payload = data.get("data") if isinstance(data, dict) else data
            price = _coerce_float(
                payload,
                ["price", "last", "last_price", "close", "close_price", "ap", "bp"],
            )
            if price is not None:
                return price
        return None

    def extract_expiries(
        self, items: List[dict], *, max_expiries: Optional[int] = None
    ) -> List[str]:
        expiries = sorted({e for raw in items if (e := _extract_expiry(raw))})
        if max_expiries:
            return expiries[:max_expiries]
        return expiries

    def quotes_from_chain(self, symbol: str, items: List[dict]) -> List[OptionQuote]:
        quotes: List[OptionQuote] = []
        for raw in items:
            expiry = _extract_expiry(raw)
            if not expiry:
                continue
            quotes.append(_normalise_quote(symbol, expiry, raw))
        return quotes

    def quotes_for_expiry(
        self, symbol: str, items: List[dict], expiry: str
    ) -> List[OptionQuote]:
        quotes: List[OptionQuote] = []
        for raw in items:
            raw_expiry = _extract_expiry(raw)
            if raw_expiry and raw_expiry != expiry:
                continue
            quotes.append(_normalise_quote(symbol, raw_expiry or expiry, raw))
        return quotes

    # --- HTTP internals ---

    def _auth_headers(self) -> Dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self.min_request_interval_seconds:
            with self._rate_limit_lock:
                now = time.monotonic()
                wait_for = self.min_request_interval_seconds - (now - self._last_request_ts)
                if wait_for > 0:
                    time.sleep(wait_for)
                self._last_request_ts = time.monotonic()
        url = f"{self.base_url}{path}"
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._session.request(
                    method,
                    url,
                    params=params,
                    headers=self._auth_headers(),
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                if attempt == self.max_retries:
                    raise
                LOGGER.warning("HTTP %s %s failed (%s), retrying", method, path, exc)
                time.sleep(0.5 * attempt)
                continue
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                LOGGER.warning("HTTP %s %s got %s, retrying", method, path, resp.status_code)
                time.sleep(0.5 * attempt)
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Failed to call {path} after {self.max_retries} attempts")


# ------------------------------------------------------------------
# Response parsing helpers
# ------------------------------------------------------------------

def _extract_items(data: Any) -> List[Dict[str, Any]]:
    def _snapshots_to_items(snapshots: Dict[str, Any]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for sym, snapshot in snapshots.items():
            if not isinstance(snapshot, dict):
                continue
            enriched = dict(snapshot)
            enriched.setdefault("symbol", sym)
            items.append(enriched)
        return items

    if isinstance(data, dict):
        if "snapshots" in data and isinstance(data["snapshots"], dict):
            return _snapshots_to_items(data["snapshots"])
        if "data" in data:
            payload = data["data"]
            if isinstance(payload, dict) and "snapshots" in payload and isinstance(payload["snapshots"], dict):
                return _snapshots_to_items(payload["snapshots"])
            data = payload
    payload = data
    if isinstance(payload, dict):
        for key in ("items", "options", "contracts", "option_contracts", "results"):
            if key in payload and isinstance(payload[key], list):
                return [item for item in payload[key] if isinstance(item, dict)]
        return []
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _extract_expiry(raw: Dict[str, Any]) -> str:
    expiry = _coerce_str(
        raw,
        ["expiration_date", "expiration-date", "expiry", "exp_date", "expiration"],
    )
    if expiry:
        return expiry
    symbol = _coerce_str(raw, ["symbol", "S", "option_symbol", "contract_symbol", "id"])
    parsed = parse_opra_symbol(symbol) if symbol else None
    return parsed[1] if parsed else ""


def _normalise_quote(symbol: str, expiry: str, raw: Dict[str, Any]) -> OptionQuote:
    option_symbol = _coerce_str(
        raw, ["symbol", "S", "option_symbol", "contract_symbol", "id"]
    )
    option_type_raw = _coerce_str(
        raw, ["option_type", "type", "right", "put_call", "call_put", "side"]
    )
    parsed = parse_opra_symbol(option_symbol or "") if option_symbol else None

    option_type = None
    if option_type_raw:
        option_type = "call" if option_type_raw.lower().startswith("c") else "put"
    if option_type is None and parsed:
        option_type = parsed[3]
    if option_type is None:
        option_type = "put"

    strike = _coerce_float(raw, ["strike", "strike_price", "strike-price"], default=0.0) or 0.0
    if not strike and parsed:
        strike = parsed[2]
    if not expiry and parsed:
        expiry = parsed[1]

    bid = _coerce_float(raw, ["bid", "bid_price", "bp", "best_bid"], default=0.0) or 0.0
    ask = _coerce_float(raw, ["ask", "ask_price", "ap", "best_ask"], default=0.0) or 0.0
    last = _coerce_float(raw, ["last", "last_price", "trade_price", "p"], default=0.0) or 0.0
    mark = _coerce_float(raw, ["mark", "mid", "midpoint"], default=0.0) or 0.0
    volume = int(_coerce_float(raw, ["volume", "trade_volume", "v", "dv"], default=0.0) or 0.0)
    open_interest = int(_coerce_float(raw, ["open_interest", "openInterest", "oi"], default=0.0) or 0.0)
    bid_size = _coerce_float(raw, ["bid_size", "bidSize", "bs"], default=None)
    ask_size = _coerce_float(raw, ["ask_size", "askSize", "as"], default=None)

    return OptionQuote(
        ticker=symbol,
        expiry=expiry,
        strike=strike,
        option_type=option_type,
        bid=bid,
        ask=ask,
        last=last,
        mark=mark,
        volume=volume,
        open_interest=open_interest,
        bid_size=int(bid_size) if bid_size is not None else None,
        ask_size=int(ask_size) if ask_size is not None else None,
        symbol=option_symbol,
    )


def _collect_payload_dicts(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources = [raw]
    for key in (
        "latestQuote", "latest_quote", "quote",
        "latestTrade", "latest_trade", "trade",
        "snapshot", "dailyBar", "daily_bar", "greeks",
    ):
        val = raw.get(key)
        if isinstance(val, dict):
            sources.append(val)
    return sources


def _coerce_float(
    raw: Any,
    keys: Iterable[str],
    default: Optional[float] = None,
) -> Optional[float]:
    if not isinstance(raw, dict):
        return default
    sources = _collect_payload_dicts(raw)
    for source in sources:
        for key in keys:
            if key not in source:
                continue
            val = source[key]
            if val in (None, ""):
                continue
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return default


def _coerce_str(raw: Any, keys: Iterable[str]) -> str:
    if not isinstance(raw, dict):
        return ""
    sources = _collect_payload_dicts(raw)
    for source in sources:
        for key in keys:
            val = source.get(key)
            if isinstance(val, str) and val:
                return val
    return ""
