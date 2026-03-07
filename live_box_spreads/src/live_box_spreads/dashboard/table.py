"""Top opportunities data table."""
from __future__ import annotations

import polars as pl


def make_top_table(df: pl.DataFrame, n: int = 10) -> list[dict]:
    """Build top lend/borrow opportunities for display in a DataTable.

    Best lending  = highest lend_rate  (sorted descending)
    Best borrowing = lowest borrow_rate (sorted ascending)
    """
    if df.is_empty():
        return []
    cols = [
        "ticker", "expiry", "kl", "ks", "width",
        "mid_rate", "lend_rate", "borrow_rate", "min_leg_volume",
    ]
    available = [c for c in cols if c in df.columns]
    if not available:
        return []

    top_lend = (
        df.filter(pl.col("lend_rate").is_not_nan() & (pl.col("lend_rate") > 0))
        .sort("lend_rate", descending=True)
        .head(n)
        .select(available)
    )
    top_borrow = (
        df.filter(pl.col("borrow_rate").is_not_nan() & (pl.col("borrow_rate") < 0))
        .sort("borrow_rate", descending=True)  # least negative = cheapest borrowing
        .head(n)
        .select(available)
    )
    combined = pl.concat([top_lend, top_borrow])

    for col in ["mid_rate", "lend_rate", "borrow_rate"]:
        if col in combined.columns:
            combined = combined.with_columns(
                (pl.col(col) * 100)
                .round(2)
                .cast(pl.Utf8)
                .str.replace(r"$", "%")
                .alias(col)
            )
    return combined.to_dicts()
