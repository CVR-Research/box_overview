"""Pure data transformation functions for option quotes.

All functions operate on Polars DataFrames — no Pandas anywhere.
"""
from __future__ import annotations

from typing import List, Optional

import polars as pl

from live_box_spreads.core.types import OptionQuote


def quotes_to_frame(quotes: List[OptionQuote]) -> pl.DataFrame:
    """Convert a list of OptionQuote objects to a Polars DataFrame with a mid column."""
    if not quotes:
        return pl.DataFrame()
    frame = pl.DataFrame(
        {
            "ticker": [q.ticker for q in quotes],
            "expiry": [q.expiry for q in quotes],
            "strike": [float(q.strike) for q in quotes],
            "type": [q.option_type.lower() for q in quotes],
            "bid": [float(q.bid) for q in quotes],
            "ask": [float(q.ask) for q in quotes],
            "last": [float(q.last) for q in quotes],
            "mark": [float(q.mark) for q in quotes],
            "volume": [int(q.volume) for q in quotes],
            "open_interest": [int(q.open_interest) for q in quotes],
        }
    )
    # Compute mid with fallback chain: mid -> mark -> last
    frame = frame.with_columns(
        pl.when((pl.col("bid") > 0) & (pl.col("ask") > 0))
        .then((pl.col("bid") + pl.col("ask")) / 2.0)
        .otherwise(None)
        .alias("mid")
    )
    frame = frame.with_columns(
        pl.when(pl.col("mid").is_null())
        .then(pl.col("mark"))
        .otherwise(pl.col("mid"))
        .alias("mid")
    )
    frame = frame.with_columns(
        pl.when(pl.col("mid").is_null())
        .then(pl.col("last"))
        .otherwise(pl.col("mid"))
        .alias("mid")
    )
    return frame


def pivot_calls_puts(frame: pl.DataFrame) -> pl.DataFrame:
    """Pivot a quotes frame into wide format: one row per strike with call_*/put_* columns."""
    if frame.is_empty():
        return frame
    calls = (
        frame.filter(pl.col("type") == "call")
        .select(
            "strike",
            pl.col("bid").alias("call_bid"),
            pl.col("ask").alias("call_ask"),
            pl.col("mid").alias("call_mid"),
            pl.col("volume").alias("call_volume"),
            pl.col("open_interest").alias("call_open_interest"),
        )
        .sort("strike")
    )
    puts = (
        frame.filter(pl.col("type") == "put")
        .select(
            "strike",
            pl.col("bid").alias("put_bid"),
            pl.col("ask").alias("put_ask"),
            pl.col("mid").alias("put_mid"),
            pl.col("volume").alias("put_volume"),
            pl.col("open_interest").alias("put_open_interest"),
        )
        .sort("strike")
    )
    return calls.join(puts, on="strike", how="inner")


def derive_spot_from_chain(quotes: List[OptionQuote]) -> Optional[float]:
    """Derive implied forward/spot price from put-call parity.

    At any strike K: Forward = K + Call_mid - Put_mid.
    The most reliable estimate comes from the ATM strike where |C - P| is minimized.
    """
    by_strike: dict[float, dict[str, float]] = {}
    for q in quotes:
        k = q.strike
        mid = (q.bid + q.ask) / 2.0 if q.bid > 0 and q.ask > 0 else q.mark or q.last
        if mid <= 0:
            continue
        if k not in by_strike:
            by_strike[k] = {}
        by_strike[k][q.option_type] = mid

    candidates = []
    for k, sides in by_strike.items():
        if "call" in sides and "put" in sides:
            c, p = sides["call"], sides["put"]
            forward = k + c - p
            candidates.append((abs(c - p), forward, k))

    if not candidates:
        return None

    candidates.sort(key=lambda x: x[0])
    _, forward, _ = candidates[0]
    return forward if forward > 0 else None
