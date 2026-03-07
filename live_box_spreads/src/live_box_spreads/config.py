"""Unified configuration and credential loading."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass(frozen=True)
class Config:
    """Immutable application configuration loaded from YAML."""

    tickers: List[str] = field(default_factory=lambda: ["SPX"])
    expiries_per_ticker: int = 2
    min_volume: int = 0
    min_open_interest: int = 0
    min_strike_gap: float = 0.0
    max_strike_gap: float = float("inf")
    price_selector: str = "mid"
    snapshot_storage: Path = Path("data/live_snapshots")
    update_interval_seconds: int = 60
    max_snapshots_retained: int = 200
    surface_history_minutes: int = 30
    max_abs_rate: float = 2.0

    @classmethod
    def from_dict(cls, payload: dict) -> Config:
        return cls(
            tickers=list(payload.get("tickers", ["SPX"])),
            expiries_per_ticker=int(payload.get("expiries_per_ticker", 2)),
            min_volume=int(payload.get("min_volume", 0)),
            min_open_interest=int(payload.get("min_open_interest", 0)),
            min_strike_gap=float(payload.get("min_strike_gap", 0.0)),
            max_strike_gap=float(payload.get("max_strike_gap", float("inf"))),
            price_selector=str(payload.get("price_selector", "mid")),
            snapshot_storage=Path(payload.get("snapshot_storage", "data/live_snapshots")),
            update_interval_seconds=int(payload.get("update_interval_seconds", 60)),
            max_snapshots_retained=int(payload.get("max_snapshots_retained", 200)),
            surface_history_minutes=int(payload.get("surface_history_minutes", 30)),
            max_abs_rate=float(payload.get("max_abs_rate", 2.0)),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)


def find_config(start: Path | None = None) -> Path:
    """Locate config.yaml by walking up from *start* (default: CWD)."""
    candidates = [
        Path.cwd() / "config.yaml",
        Path(__file__).resolve().parent.parent.parent / "config.yaml",
    ]
    if start is not None:
        candidates.insert(0, start / "config.yaml")
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("config.yaml not found")


def load_credentials() -> tuple[str, str]:
    """Load Alpaca API credentials from environment. Single canonical source."""
    api_key = os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_API_KEY_ID")
    api_secret = os.environ.get("ALPACA_API_SECRET") or os.environ.get("ALPACA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_API_SECRET env vars required")
    return api_key, api_secret
