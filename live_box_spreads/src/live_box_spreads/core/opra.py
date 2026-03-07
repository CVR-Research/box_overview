"""Single canonical OPRA symbol parser. Eliminates duplication."""
from __future__ import annotations

import re
from datetime import date
from typing import Optional, Tuple

OPRA_PATTERN = re.compile(
    r"^(?P<root>[A-Z0-9]{1,6})(?P<date>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$"
)


def parse_opra_symbol(symbol: str) -> Optional[Tuple[str, str, float, str]]:
    """Parse an OPRA symbol into (root, expiry_iso, strike, 'call'|'put').

    Returns None if the symbol does not match OPRA format.
    Example: "SPX260118C04000000" -> ("SPX", "2026-01-18", 4000.0, "call")
    """
    cleaned = symbol.replace(" ", "").upper()
    match = OPRA_PATTERN.match(cleaned)
    if not match:
        return None
    root = match.group("root").strip()
    yymmdd = match.group("date")
    year = 2000 + int(yymmdd[0:2])
    month = int(yymmdd[2:4])
    day = int(yymmdd[4:6])
    try:
        expiry = date(year, month, day).isoformat()
    except ValueError:
        return None
    option_type = "call" if match.group("cp") == "C" else "put"
    strike = int(match.group("strike")) / 1000.0
    return root, expiry, strike, option_type


def extract_underlying(symbol: str) -> str:
    """Extract underlying ticker from an OPRA symbol. Falls back to the symbol itself."""
    parsed = parse_opra_symbol(symbol)
    if parsed:
        return parsed[0]
    return symbol
