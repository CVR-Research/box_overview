"""Stateless spread construction engine.

Converts option quotes into box spread records with implied rates
using per-strike synthetic forward prices (SFL / SFS).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from live_box_spreads.config import Config
from live_box_spreads.core.types import OptionQuote
from live_box_spreads.engine.transforms import pivot_calls_puts, quotes_to_frame

YEAR_SECONDS = 365.25 * 24 * 60 * 60


def parse_expiry(expiry: str) -> datetime:
    """Parse an expiry date string to a UTC datetime."""
    try:
        return datetime.fromisoformat(expiry).replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _calc_lend_rate(payoff: float, cost: float, tte: float) -> float:
    """Lend rate (positive): annualised return from buying the box.

    You pay *cost* today, receive *payoff* at expiry.
    Valid only when 0 < cost < payoff (paying < $1 to get $1 back).
    """
    if cost <= 0 or cost >= payoff or tte <= 0:
        return float("nan")
    try:
        return (payoff / cost) ** (1.0 / tte) - 1.0
    except OverflowError:
        return float("nan")


def _calc_borrow_rate(payoff: float, proceeds: float, tte: float) -> float:
    """Borrow rate (negative): annualised cost of selling the box.

    You receive *proceeds* today, owe *payoff* at expiry.
    Valid only when 0 < proceeds < payoff (getting < $1, owing $1).
    Returns a negative number (cost of capital).
    """
    if proceeds <= 0 or proceeds >= payoff or tte <= 0:
        return float("nan")
    try:
        return 1.0 - (payoff / proceeds) ** (1.0 / tte)
    except OverflowError:
        return float("nan")


class SpreadBuilder:
    """Builds box spread DataFrames from option quotes."""

    def __init__(self, config: Config) -> None:
        self.config = config

    def build_from_quotes(
        self,
        ticker: str,
        expiry: str,
        spot: Optional[float],
        quotes: List[OptionQuote],
        snapshot_time: datetime,
    ) -> pl.DataFrame:
        """Full pipeline: quotes -> frame -> pivot -> pair enumerate -> spreads."""
        merged = pivot_calls_puts(quotes_to_frame(quotes))
        if merged.is_empty():
            return pl.DataFrame()
        expiry_dt = parse_expiry(expiry)
        return self._enumerate_pairs(ticker, expiry_dt, spot, merged, snapshot_time)

    def _enumerate_pairs(
        self,
        ticker: str,
        expiry_dt: datetime,
        spot: Optional[float],
        merged: pl.DataFrame,
        snapshot_time: datetime,
    ) -> pl.DataFrame:
        """O(n^2) strike pair enumeration with liquidity and gap filters."""
        config = self.config
        valid_rows = [
            row for row in merged.iter_rows(named=True)
            if _is_liquid(row, config.min_volume, config.min_open_interest)
        ]
        if len(valid_rows) < 2:
            return pl.DataFrame()

        tte_years = max(
            (expiry_dt - snapshot_time).total_seconds() / YEAR_SECONDS, 0.0001
        )

        # --- Step 1: per-strike synthetic forward prices ---
        # SFL(K) = cost to go long  the forward = Ask(C) - Bid(P)
        # SFS(K) = proceeds to go short the forward = Bid(C) - Ask(P)
        # SF_mid(K) = theoretical mid = Mid(C) - Mid(P)
        for row in valid_rows:
            row["sfl"] = float(row["call_ask"]) - float(row["put_bid"])
            row["sfs"] = float(row["call_bid"]) - float(row["put_ask"])
            row["sf_mid"] = float(row["call_mid"]) - float(row["put_mid"])

        # --- Step 2: enumerate strike pairs ---
        records: List[Dict] = []
        max_rate = config.max_abs_rate
        _nan = float("nan")

        for idx, lo in enumerate(valid_rows):
            for hi in valid_rows[idx + 1:]:
                kl = float(lo["strike"])
                ks = float(hi["strike"])
                width = ks - kl
                if width < config.min_strike_gap or width > config.max_strike_gap:
                    continue

                payoff = width

                # Box prices from synthetic forwards
                lend_cost = lo["sfl"] - hi["sfs"]        # long box: buy fwd@kl, sell fwd@ks
                borrow_proceeds = lo["sfs"] - hi["sfl"]  # short box: sell fwd@kl, buy fwd@ks
                mid_cost = lo["sf_mid"] - hi["sf_mid"]    # theoretical

                # Rates: lend > 0, borrow < 0
                mid_rate = _calc_lend_rate(payoff, mid_cost, tte_years)
                lend_rate = _calc_lend_rate(payoff, lend_cost, tte_years)
                borrow_rate = _calc_borrow_rate(payoff, borrow_proceeds, tte_years)

                # Skip if mid rate is invalid
                if mid_rate != mid_rate or mid_rate > max_rate:
                    continue
                # Cap extreme rates
                if lend_rate == lend_rate and lend_rate > max_rate:
                    lend_rate = _nan
                if borrow_rate == borrow_rate and borrow_rate < -max_rate:
                    borrow_rate = _nan

                record = {
                    "ticker": ticker,
                    "expiry": expiry_dt.date().isoformat(),
                    "snapshot_time": snapshot_time.isoformat(),
                    "kl": kl,
                    "ks": ks,
                    "width": width,
                    "mid_strike": (kl + ks) / 2.0,
                    "tte_years": tte_years,
                    "payoff": payoff,
                    "mid_cost": mid_cost,
                    "lend_cost": lend_cost,
                    "borrow_proceeds": borrow_proceeds,
                    "mid_rate": mid_rate,
                    "lend_rate": lend_rate,
                    "borrow_rate": borrow_rate,
                    "spot_price": spot or np.nan,
                    "moneyness_kl": (kl / spot) if spot else np.nan,
                    "moneyness_ks": (ks / spot) if spot else np.nan,
                    "moneyness_mid": ((kl + ks) / 2.0 / spot) if spot else np.nan,
                    "call_kl_volume": int(lo["call_volume"]),
                    "call_ks_volume": int(hi["call_volume"]),
                    "put_kl_volume": int(lo["put_volume"]),
                    "put_ks_volume": int(hi["put_volume"]),
                    "call_kl_open_interest": int(lo["call_open_interest"]),
                    "call_ks_open_interest": int(hi["call_open_interest"]),
                    "put_kl_open_interest": int(lo["put_open_interest"]),
                    "put_ks_open_interest": int(hi["put_open_interest"]),
                }
                record["min_leg_volume"] = min(
                    record["call_kl_volume"],
                    record["call_ks_volume"],
                    record["put_kl_volume"],
                    record["put_ks_volume"],
                )
                record["total_leg_volume"] = (
                    record["call_kl_volume"]
                    + record["call_ks_volume"]
                    + record["put_kl_volume"]
                    + record["put_ks_volume"]
                )
                records.append(record)

        if not records:
            return pl.DataFrame()
        return pl.DataFrame(records)


def _is_liquid(row: dict, min_vol: int, min_oi: int) -> bool:
    """Check if all legs of a strike row meet minimum liquidity thresholds."""
    vols = [row.get("call_volume", 0), row.get("put_volume", 0)]
    ois = [row.get("call_open_interest", 0), row.get("put_open_interest", 0)]
    return all(v >= min_vol for v in vols) and all(oi >= min_oi for oi in ois)
