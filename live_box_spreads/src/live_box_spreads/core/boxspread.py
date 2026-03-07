"""Core data model for box spreads.

Immutable representation of index-option box spreads with financial math
for cost calculation and implied rate extraction.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from math import copysign
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class LendBorrow(IntEnum):
    BORROW = -1  # net credit
    FLAT = 0
    LEND = +1  # net debit

    def __str__(self) -> str:
        if self is LendBorrow.LEND:
            return "lend"
        if self is LendBorrow.BORROW:
            return "borrow"
        return "flat"


@dataclass(slots=True, frozen=True)
class BoxSpread:
    """Immutable box spread with derived cost and implied rate."""

    ticker: str
    expiry: datetime
    kl: float  # lower strike
    ks: float  # upper strike
    prices: Dict[str, float]  # call_kl, call_ks, put_kl, put_ks
    tte: float  # time-to-expiry in years

    # derived (set in __post_init__)
    net_cost: float = 0.0
    implied_rate: float = 0.0
    lend_borrow: LendBorrow = LendBorrow.FLAT

    def __post_init__(self) -> None:
        cost = self._calc_cost()
        rate = self._calc_implied_rate(cost)
        lb = (
            LendBorrow.LEND
            if cost > 0
            else LendBorrow.BORROW
            if cost < 0
            else LendBorrow.FLAT
        )
        object.__setattr__(self, "net_cost", cost)
        object.__setattr__(self, "implied_rate", rate)
        object.__setattr__(self, "lend_borrow", lb)

    # --- static math (numba-friendly) ---

    @staticmethod
    def calc_cost(
        call_kl: float, call_ks: float, put_kl: float, put_ks: float
    ) -> float:
        return (call_kl + put_ks) - (put_kl + call_ks)

    @staticmethod
    def calc_implied_rate(payoff: float, cost: float, tte: float) -> float:
        if cost == 0.0 or payoff == 0.0 or tte <= 0:
            return 0.0
        try:
            rate = (payoff / abs(cost)) ** (1.0 / tte) - 1.0
        except OverflowError:
            return 0.0
        return copysign(rate, cost)

    # --- internals ---

    def _calc_cost(self) -> float:
        p = self.prices
        return BoxSpread.calc_cost(p["call_kl"], p["call_ks"], p["put_kl"], p["put_ks"])

    def _calc_implied_rate(self, cost: float) -> float:
        payoff = abs(self.ks - self.kl)
        return BoxSpread.calc_implied_rate(payoff, cost, self.tte)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "expiry": self.expiry,
            "kl": self.kl,
            "ks": self.ks,
            "net_cost": self.net_cost,
            "implied_rate": self.implied_rate,
            "lend_borrow": str(self.lend_borrow),
        }

    # --- factory loaders ---

    @classmethod
    def from_snapshot_row(cls, row: "pd.Series", tte: float) -> BoxSpread:
        """Create from a pivoted snapshot row with call_kl/call_ks/put_kl/put_ks columns."""

        def _get(field: str):
            try:
                return row[field]
            except Exception:
                return getattr(row, field)

        prices = {
            "call_kl": _get("call_kl"),
            "call_ks": _get("call_ks"),
            "put_kl": _get("put_kl"),
            "put_ks": _get("put_ks"),
        }
        return cls(_get("ticker"), _get("expiry"), _get("kl"), _get("ks"), prices, tte)

    @classmethod
    def batch_from_snapshot(cls, frame: "pd.DataFrame", *, tte: float) -> List[BoxSpread]:
        """Create BoxSpreads from all rows in a pivoted DataFrame."""
        return [cls.from_snapshot_row(r, tte) for r in frame.itertuples(index=False)]

    def __str__(self) -> str:
        dir_s = str(self.lend_borrow)
        return (
            f"{self.ticker} {self.expiry:%Y-%m-%d}  box[{self.kl}/{self.ks}]  {dir_s}  "
            f"net={'debit' if self.net_cost > 0 else 'credit'} {abs(self.net_cost):.2f}  "
            f"r={self.implied_rate:+.2%}"
        )

    __repr__ = __str__
