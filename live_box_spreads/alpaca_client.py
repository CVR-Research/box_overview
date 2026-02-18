"""Thin wrapper around the Alpaca Market Data REST API.

The client keeps a session alive, retries transient failures, and exposes a
small API tailored for the live box-spread pipeline.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = os.environ.get("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets")
DEFAULT_FEED = os.environ.get("ALPACA_DATA_FEED", "opra")


@dataclass
class OptionQuote:
    ticker: str
    expiry: str
    strike: float
    option_type: str  # "call" | "put"
    bid: float
    ask: float
    last: float
    mark: float
    volume: int
    open_interest: int
    bid_size: Optional[int] = None
    ask_size: Optional[int] = None
    symbol: Optional[str] = None


class AlpacaMarketDataClient:
    """Simple authenticated wrapper over the Alpaca market data endpoints."""

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        feed: str = DEFAULT_FEED,
        timeout: float = 15.0,
        max_retries: int = 3,
    ) -> None:
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.feed = feed
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()

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
            except requests.RequestException as exc:  # pragma: no cover - IO error
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
    # Public API
    # ------------------------------------------------------------------
    def get_expiries(self, symbol: str) -> List[str]:
        """Return sorted expiry dates (YYYY-MM-DD)."""
        params = {"feed": self.feed}
        paths = (
            ("/v1beta1/options/chain", {"underlying_symbol": symbol, **params}),
            ("/v1beta1/options/chain", {"symbol": symbol, **params}),
            ("/v2/options/contracts", {"underlying_symbols": symbol, **params, "limit": 10000}),
        )
        last_exc: Optional[Exception] = None
        for path, path_params in paths:
            try:
                data = self._request("GET", path, params=path_params)
            except Exception as exc:  # pragma: no cover - best effort
                last_exc = exc
                continue
            items = _extract_items(data)
            expiries = _extract_expiries(items)
            if expiries:
                return sorted(expiries)
        raise RuntimeError(f"Unable to fetch expiries for {symbol}: {last_exc}")

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
            except Exception:  # pragma: no cover - fall through
                continue
            payload = data.get("data") if isinstance(data, dict) else data
            price = _coerce_float_from_payload(
                payload,
                [
                    "price",
                    "last",
                    "last_price",
                    "close",
                    "close_price",
                    "ap",
                    "bp",
                ],
            )
            if price is not None:
                return price
        return None

    def get_option_quotes(self, symbol: str, expiry: str) -> List[OptionQuote]:
        params = {"feed": self.feed}
        paths = (
            ("/v1beta1/options/chain", {"underlying_symbol": symbol, "expiration_date": expiry, **params}),
            ("/v1beta1/options/chain", {"symbol": symbol, "expiration_date": expiry, **params}),
            (f"/v1beta1/options/snapshots/{symbol}", params),
            ("/v2/options/contracts", {"underlying_symbols": symbol, "expiration_date": expiry, **params, "limit": 10000}),
        )
        last_exc: Optional[Exception] = None
        for path, path_params in paths:
            try:
                data = self._request("GET", path, params=path_params)
            except Exception as exc:  # pragma: no cover - try fallback path
                last_exc = exc
                continue
            items = _extract_items(data)
            quotes = []
            for raw in items:
                raw_expiry = _coerce_str_from_payload(raw, ["expiration_date", "expiration-date", "expiry", "exp_date", "expiration"])
                if raw_expiry and raw_expiry != expiry:
                    continue
                quotes.append(self._normalise_quote(symbol, raw_expiry or expiry, raw))
            if quotes:
                return quotes
        raise RuntimeError(f"Unable to fetch quotes for {symbol} {expiry}: {last_exc}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _normalise_quote(self, symbol: str, expiry: str, raw: Dict[str, Any]) -> OptionQuote:
        option_symbol = _coerce_str_from_payload(
            raw,
            ["symbol", "S", "option_symbol", "contract_symbol", "id"],
        )
        option_type_raw = _coerce_str_from_payload(
            raw,
            ["option_type", "type", "right", "put_call", "call_put", "side"],
        )
        option_type = "call" if option_type_raw.lower().startswith("c") else "put"
        strike = _coerce_float_from_payload(raw, ["strike", "strike_price", "strike-price"], default=0.0) or 0.0
        bid = _coerce_float_from_payload(raw, ["bid", "bid_price", "bp", "best_bid"], default=0.0) or 0.0
        ask = _coerce_float_from_payload(raw, ["ask", "ask_price", "ap", "best_ask"], default=0.0) or 0.0
        last = _coerce_float_from_payload(raw, ["last", "last_price", "trade_price", "p"], default=0.0) or 0.0
        mark = _coerce_float_from_payload(raw, ["mark", "mid", "midpoint"], default=0.0) or 0.0
        volume = int(_coerce_float_from_payload(raw, ["volume", "trade_volume", "v", "dv"], default=0.0) or 0.0)
        open_interest = int(_coerce_float_from_payload(raw, ["open_interest", "openInterest", "oi"], default=0.0) or 0.0)
        bid_size = _coerce_float_from_payload(raw, ["bid_size", "bidSize", "bs"], default=None)
        ask_size = _coerce_float_from_payload(raw, ["ask_size", "askSize", "as"], default=None)
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


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _extract_items(data: Any) -> List[Dict[str, Any]]:
    payload = data.get("data") if isinstance(data, dict) else data
    if isinstance(payload, dict):
        for key in ("items", "options", "contracts", "results"):
            if key in payload and isinstance(payload[key], list):
                return payload[key]
        if "snapshots" in payload and isinstance(payload["snapshots"], dict):
            return [v for v in payload["snapshots"].values() if isinstance(v, dict)]
        return [payload]
    if isinstance(payload, list):
        return payload
    return []


def _extract_expiries(items: Iterable[Dict[str, Any]]) -> List[str]:
    expiries: List[str] = []
    for raw in items:
        expiry = _coerce_str_from_payload(
            raw,
            ["expiration_date", "expiration-date", "expiry", "exp_date", "expiration"],
        )
        if expiry:
            expiries.append(expiry)
    return list({e for e in expiries})


def _collect_payload_dicts(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    sources = [raw]
    for key in (
        "latestQuote",
        "latest_quote",
        "quote",
        "latestTrade",
        "latest_trade",
        "trade",
        "snapshot",
        "dailyBar",
        "daily_bar",
        "greeks",
    ):
        val = raw.get(key)
        if isinstance(val, dict):
            sources.append(val)
    return sources


def _coerce_float_from_payload(
    raw: Any,
    keys: Iterable[str],
    default: Optional[float] = None,
) -> Optional[float]:
    if isinstance(raw, dict):
        sources = _collect_payload_dicts(raw)
    else:
        return default
    for source in sources:
        for key in keys:
            if key not in source:
                continue
            val = source[key]
            if val in (None, ""):
                continue
            try:
                return float(val)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
    return default


def _coerce_str_from_payload(raw: Any, keys: Iterable[str]) -> str:
    if not isinstance(raw, dict):
        return ""
    sources = _collect_payload_dicts(raw)
    for source in sources:
        for key in keys:
            val = source.get(key)
            if isinstance(val, str) and val:
                return val
    return ""


__all__ = ["AlpacaMarketDataClient", "OptionQuote"]
