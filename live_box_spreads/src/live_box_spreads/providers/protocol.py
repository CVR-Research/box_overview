"""DataProvider protocol — abstraction over market data sources."""
from __future__ import annotations

from typing import List, Optional, Protocol, runtime_checkable

from live_box_spreads.core.types import OptionQuote


@runtime_checkable
class DataProvider(Protocol):
    """Structural interface for any market data source (Alpaca, CBOE, mock, etc.)."""

    def get_option_chain(self, symbol: str) -> List[dict]:
        """Return raw option chain snapshot items for an underlying."""
        ...

    def get_underlying_price(self, symbol: str) -> Optional[float]:
        """Return current price for the underlying, or None."""
        ...

    def extract_expiries(
        self, items: List[dict], *, max_expiries: Optional[int] = None
    ) -> List[str]:
        """Extract sorted expiry dates from chain items."""
        ...

    def quotes_from_chain(self, symbol: str, items: List[dict]) -> List[OptionQuote]:
        """Convert all raw chain items to OptionQuote objects."""
        ...

    def quotes_for_expiry(
        self, symbol: str, items: List[dict], expiry: str
    ) -> List[OptionQuote]:
        """Convert chain items for a specific expiry to OptionQuote objects."""
        ...
