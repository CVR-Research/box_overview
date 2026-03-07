"""DataSource protocol — unified interface the dashboard consumes."""
from __future__ import annotations

from typing import Protocol

import polars as pl


class DataSource(Protocol):
    """The dashboard calls only these two methods.

    Implementations can read from disk, poll REST, or stream via WebSocket.
    """

    def latest_spreads(self) -> pl.DataFrame:
        """Return the latest box spread data as a Polars DataFrame."""
        ...

    def status(self) -> dict:
        """Return a status dict for display in the dashboard status bar."""
        ...
