"""Thin wrapper around the public Tastytrade REST API.

This client keeps an authenticated session alive, retries transient
network failures, and exposes a small, typed-ish API surface tailored
for our box-spread pipeline.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import requests

LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = os.environ.get("TASTYTRADE_BASE_URL", "https://api.tastytrade.com")
SESSION_PATHS: tuple[str, ...] = (
    "/rest/public/sessions",
    "/sessions",
)


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


class TastytradeClient:
    """Simple authenticated wrapper over the REST endpoints."""

    def __init__(
        self,
        username: str,
        password: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 15.0,
        max_retries: int = 3,
    ) -> None:
        self.username = username
        self.password = password
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session_token: Optional[str] = None
        self._token_expiry: float = 0.0

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------
    def authenticate(self) -> None:
        payload = {"login": self.username, "password": self.password}
        last_error: Optional[str] = None
        for path in SESSION_PATHS:
            url = f"{self.base_url}{path}"
            try:
                resp = self._session.post(url, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:  # pragma: no cover - IO error
                last_error = str(exc)
                continue
            if resp.ok:
                token = (
                    resp.headers.get("session-token")
                    or resp.json()
                    .get("data", {})
                    .get("session-token")
                )
                if not token:
                    last_error = "Session token missing in response"
                    continue
                self._session_token = token
                # tokens generally live for 24h; be conservative
                self._token_expiry = time.time() + 60 * 60 * 6
                LOGGER.info("Authenticated via %s", path)
                return
            last_error = f"{resp.status_code}: {resp.text}"
        raise RuntimeError(f"Failed to authenticate with Tastytrade API: {last_error}")

    def _auth_headers(self) -> Dict[str, str]:
        if not self._session_token or time.time() > self._token_expiry:
            self.authenticate()
        assert self._session_token is not None
        return {
            "Authorization": self._session_token,
            "session-token": self._session_token,
        }

    # ------------------------------------------------------------------
    # Generic request helper
    # ------------------------------------------------------------------
    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        for attempt in range(1, self.max_retries + 1):
            headers = self._auth_headers()
            try:
                resp = self._session.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=headers,
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:  # pragma: no cover - IO error
                if attempt == self.max_retries:
                    raise
                LOGGER.warning("HTTP %s %s failed (%s), retrying", method, path, exc)
                time.sleep(0.5 * attempt)
                continue
            if resp.status_code == 401 and attempt < self.max_retries:
                LOGGER.info("Session expired, refreshing…")
                self.authenticate()
                continue
            resp.raise_for_status()
            return resp.json()
        raise RuntimeError(f"Failed to call {path} after {self.max_retries} attempts")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get_expiries(self, symbol: str) -> List[str]:
        """Return sorted expiry dates (YYYY-MM-DD)."""
        paths = (
            f"/rest/public/equities/option-chains/{symbol}/expirations",
            f"/option-chains/{symbol}/expirations",
        )
        last_exc: Optional[Exception] = None
        for path in paths:
            try:
                data = self._request("GET", path)
            except Exception as exc:  # pragma: no cover - best effort
                last_exc = exc
                continue
            expiries = (
                data.get("data")
                or data.get("expirations")
                or data.get("items")
                or []
            )
            if isinstance(expiries, dict):
                expiries = expiries.get("items", [])
            return sorted(str(e["expiration-date"] if isinstance(e, dict) else e) for e in expiries)
        raise RuntimeError(f"Unable to fetch expiries for {symbol}: {last_exc}")

    def get_underlying_price(self, symbol: str) -> Optional[float]:
        paths = (
            f"/rest/public/markets/quotes/{symbol}",
            f"/markets/quotes/{symbol}",
            f"/rest/public/market-metrics/{symbol}",
        )
        for path in paths:
            try:
                data = self._request("GET", path)
            except Exception:  # pragma: no cover - fall through
                continue
            payload = data.get("data") or data
            if isinstance(payload, dict) and "last" in payload:
                quote = payload
            elif isinstance(payload, dict) and "items" in payload:
                quote = payload["items"][0]
            else:
                quote = payload
            price = _coerce_float(
                quote,
                [
                    "last",
                    "last-price",
                    "mark",
                    "mark-price",
                    "close",
                    "close-price",
                ],
            )
            if price is not None:
                return price
        return None

    def get_option_quotes(self, symbol: str, expiry: str) -> List[OptionQuote]:
        paths = (
            f"/rest/public/equities/option-chains/{symbol}/quotes",
            f"/option-chains/{symbol}/quotes",
        )
        params = {"expiration-date": expiry}
        last_exc: Optional[Exception] = None
        for path in paths:
            try:
                data = self._request("GET", path, params=params)
            except Exception as exc:  # pragma: no cover - try fallback path
                last_exc = exc
                continue
            payload = data.get("data") or data
            items: Iterable[Dict[str, Any]]
            if isinstance(payload, dict) and "items" in payload:
                items = payload["items"]
            elif isinstance(payload, list):
                items = payload
            else:
                items = []
            return [self._normalise_quote(symbol, expiry, raw) for raw in items]
        raise RuntimeError(f"Unable to fetch quotes for {symbol} {expiry}: {last_exc}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _normalise_quote(
        self, symbol: str, expiry: str, raw: Dict[str, Any]
    ) -> OptionQuote:
        option_type_raw = _coerce_str(
            raw,
            ["option-type", "call-put-indicator", "type", "right"],
            default="call",
        )
        option_type = "call" if option_type_raw.lower().startswith("c") else "put"
        return OptionQuote(
            ticker=symbol,
            expiry=expiry,
            strike=_coerce_float(raw, ["strike", "strike-price", "strike-price-decimal"]) or 0.0,
            option_type=option_type,
            bid=_coerce_float(raw, ["bid", "bid-price", "best-bid"], default=0.0) or 0.0,
            ask=_coerce_float(raw, ["ask", "ask-price", "best-ask"], default=0.0) or 0.0,
            last=_coerce_float(raw, ["last", "last-price"], default=0.0) or 0.0,
            mark=_coerce_float(raw, ["mark", "mark-price", "mid"], default=0.0) or 0.0,
            volume=int(_coerce_float(raw, ["volume"], default=0.0) or 0.0),
            open_interest=int(
                _coerce_float(raw, ["open-interest", "openInterest"], default=0.0) or 0.0
            ),
            bid_size=int(_coerce_float(raw, ["bid-size", "bid-quantity"], default=0.0) or 0.0)
            if _coerce_float(raw, ["bid-size", "bid-quantity"], default=None) is not None
            else None,
            ask_size=int(_coerce_float(raw, ["ask-size", "ask-quantity"], default=0.0) or 0.0)
            if _coerce_float(raw, ["ask-size", "ask-quantity"], default=None) is not None
            else None,
        )


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _coerce_float(raw: Dict[str, Any], keys: Iterable[str], default: Optional[float] = None) -> Optional[float]:
    for key in keys:
        if key not in raw:
            continue
        val = raw[key]
        if val in (None, ""):
            continue
        try:
            return float(val)
        except (TypeError, ValueError):  # pragma: no cover - defensive
            continue
    return default


def _coerce_str(
    raw: Dict[str, Any], keys: Iterable[str], default: str = ""
) -> str:
    for key in keys:
        val = raw.get(key)
        if isinstance(val, str) and val:
            return val
    return default


__all__ = ["TastytradeClient", "OptionQuote"]
