"""Shared types used across all modules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True, slots=True)
class OptionQuote:
    """Immutable option quote. Used by all providers and the spread engine."""

    ticker: str
    expiry: str  # ISO date "YYYY-MM-DD"
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
    symbol: Optional[str] = None  # OPRA symbol if available
