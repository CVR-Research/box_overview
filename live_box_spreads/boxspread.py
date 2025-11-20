"""boxspread.py  ──  Core data model + loaders
=================================================
High‑performance, *immutable* representation of index‑option **box spreads**
plus helper loaders to auto‑construct objects from:

* CSV **snapshot rows** produced by our yfinance crawler.
* QuantConnect **OptionChain** slices in live/back‑test `OnData()`.

The financial maths remain Numba‑friendly and the public API stays minimal.
Only the loader layer differs between “offline snapshots” and “live QC”.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from math import copysign
from typing import Dict, Iterable, List, Final, TYPE_CHECKING

if TYPE_CHECKING:  # avoid heavy deps at runtime
    import pandas as pd
    from quantconnect import OptionChain  # type: ignore – only when running in QC

__all__: Final = ["BoxSpread", "LendBorrow"]


# ---------------------------------------------------------------------------
# Enum: lend / borrow / flat
# ---------------------------------------------------------------------------
class LendBorrow(IntEnum):
    BORROW = -1  # net credit
    FLAT = 0
    LEND = +1  # net debit

    def __str__(self):
        return (
            "lend"
            if self is LendBorrow.LEND
            else "borrow"
            if self is LendBorrow.BORROW
            else "flat"
        )


# ---------------------------------------------------------------------------
# BoxSpread dataclass (immutable, slot‑based)
# ---------------------------------------------------------------------------
@dataclass(slots=True, frozen=True)
class BoxSpread:
    ticker: str
    expiry: datetime
    kl: float  # strike of SLF
    ks: float  # strike of SSF
    prices: Dict[
        str, float
    ]  # expects call_kl, call_ks, put_kl, put_ks (bid/ask/mid chosen by caller)
    tte: float  # time‑to‑expiry in *years*

    # derived
    net_cost: float = 0.0
    implied_rate: float = 0.0
    lend_borrow: LendBorrow = LendBorrow.FLAT

    # ---------------- construction ----------------
    def __post_init__(self):
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

    # ---------------- maths (static, numba‑ready) ----------------
    @staticmethod
    def calc_cost(
        call_kl: float, call_ks: float, put_kl: float, put_ks: float
    ) -> float:
        return (call_kl + put_ks) - (put_kl + call_ks)

    @staticmethod
    def calc_implied_rate(payoff: float, cost: float, tte: float) -> float:
        if cost == 0.0 or payoff == 0.0 or tte <= 0:
            return 0.0
        rate = (payoff / abs(cost)) ** (1.0 / tte) - 1.0
        return copysign(rate, cost)

    # ---------------- internals ----------------
    def _calc_cost(self) -> float:
        p = self.prices
        return BoxSpread.calc_cost(p["call_kl"], p["call_ks"], p["put_kl"], p["put_ks"])

    def _calc_implied_rate(self, cost: float) -> float:
        payoff = abs(self.ks - self.kl)
        return BoxSpread.calc_implied_rate(payoff, cost, self.tte)

    # ---------------- public helpers ----------------
    def to_dict(self):
        return {
            "ticker": self.ticker,
            "expiry": self.expiry,
            "kl": self.kl,
            "ks": self.ks,
            "net_cost": self.net_cost,
            "implied_rate": self.implied_rate,
            "lend_borrow": str(self.lend_borrow),
        }

    # ---------------------------------------------------------------------
    # === Factory Loaders ==================================================
    # ---------------------------------------------------------------------
    @classmethod
    def from_snapshot_row(cls, row: "pd.Series", tte: float) -> "BoxSpread":
        """Create a BoxSpread from **one row** in the CSV snapshot.

        Expected columns:
            ticker, expiry, type (call/put), strike, bid, ask, mid (optional)
        This loader assumes the caller has pivoted bid/ask/mid into the four
        keys needed for `prices`.
        """
        prices = {
            "call_kl": row["call_kl"],
            "call_ks": row["call_ks"],
            "put_kl": row["put_kl"],
            "put_ks": row["put_ks"],
        }
        return cls(row["ticker"], row["expiry"], row["kl"], row["ks"], prices, tte)

    @classmethod
    def batch_from_snapshot(
        cls, frame: "pd.DataFrame", *, tte: float
    ) -> List["BoxSpread"]:
        """Vectorised creator: turn entire pivoted frame into objects."""
        return [cls.from_snapshot_row(r, tte) for _, r in frame.iterrows()]

    # ----- QuantConnect loader (works only inside QC runtime) -----
    @classmethod
    def from_qc_chain(
        cls, ticker: str, chain: "OptionChain", now: datetime, *, price_selector="mid"
    ) -> List["BoxSpread"]:
        """Convert a QC OptionChain slice to *many* BoxSpreads.

        * Builds a mini lookup of (strike→option contract) by right.
        * For each (kl, ks) pair constructs a spread with bid/ask/mid chosen by
          ``price_selector`` ("bid" | "ask" | "mid").
        * Returns a **list** of BoxSpread objects.
        """
        if price_selector not in {"bid", "ask", "mid"}:
            raise ValueError("price_selector must be 'bid', 'ask', or 'mid'")

        tte_years = (chain.Expiry.date() - now.date()).days / 365.0
        calls = {c.Strike: c for c in chain if c.Right.name == "Call"}
        puts = {p.Strike: p for p in chain if p.Right.name == "Put"}
        strikes = sorted(set(calls) & set(puts))
        spreads: List[BoxSpread] = []

        for i, kl in enumerate(strikes):
            for ks in strikes[i + 1:]:
                c_kl, c_ks = calls[kl], calls[ks]
                p_kl, p_ks = puts[kl], puts[ks]

                get = {
                    "bid": lambda oc: oc.BidPrice,
                    "ask": lambda oc: oc.AskPrice,
                    "mid": lambda oc: (oc.BidPrice + oc.AskPrice) / 2.0,
                }[price_selector]

                prices = {
                    "call_kl": get(c_kl),
                    "call_ks": get(c_ks),
                    "put_kl": get(p_kl),
                    "put_ks": get(p_ks),
                }
                spreads.append(
                    cls(ticker, chain.Expiry, kl, ks, prices, tte_years))

        return spreads

    # ---------------- nice repr ----------------
    def __str__(self):  # pragma: no cover
        dir_s = str(self.lend_borrow)
        return (
            f"{self.ticker} {self.expiry:%Y-%m-%d}  box[{self.kl}/{self.ks}]  {dir_s}  "
            f"net={'debit' if self.net_cost > 0 else 'credit'} {abs(self.net_cost):.2f}  "
            f"r={self.implied_rate:+.2%}"
        )

    __repr__ = __str__
