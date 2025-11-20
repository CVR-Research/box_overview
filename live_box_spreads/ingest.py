from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl
import yaml

from boxspread import BoxSpread
from tastytrade_client import OptionQuote, TastytradeClient

LOGGER = logging.getLogger("live_box_spreads.ingest")
YEAR_SECONDS = 365.25 * 24 * 60 * 60
CONFIG_PATH = Path(__file__).with_name("config.yaml")


@dataclass
class Config:
    tickers: List[str]
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

    @classmethod
    def from_dict(cls, payload: Dict) -> "Config":
        snapshot_storage = Path(payload.get("snapshot_storage", "data/live_snapshots"))
        return cls(
            tickers=list(payload.get("tickers", [])),
            expiries_per_ticker=int(payload.get("expiries_per_ticker", 2)),
            min_volume=int(payload.get("min_volume", 0)),
            min_open_interest=int(payload.get("min_open_interest", 0)),
            min_strike_gap=float(payload.get("min_strike_gap", 0.0)),
            max_strike_gap=float(payload.get("max_strike_gap", float("inf"))),
            price_selector=str(payload.get("price_selector", "mid")),
            snapshot_storage=snapshot_storage,
            update_interval_seconds=int(payload.get("update_interval_seconds", 60)),
            max_snapshots_retained=int(payload.get("max_snapshots_retained", 200)),
            surface_history_minutes=int(payload.get("surface_history_minutes", 30)),
        )


def load_config(path: Path = CONFIG_PATH) -> Config:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return Config.from_dict(data)


def ensure_env_credentials() -> tuple[str, str]:
    username = os.environ.get("TT_USERNAME") or os.environ.get("TASTY_USERNAME")
    password = os.environ.get("TT_PASSWORD") or os.environ.get("TASTY_PASSWORD")
    if not username or not password:
        raise RuntimeError("TT_USERNAME and TT_PASSWORD env vars are required")
    return username, password


def parse_expiry(expiry: str) -> datetime:
    try:
        return datetime.fromisoformat(expiry).replace(tzinfo=timezone.utc)
    except ValueError:
        return datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def quotes_to_frame(quotes: List[OptionQuote]) -> pl.DataFrame:
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
    merged = calls.join(puts, on="strike", how="inner")
    return merged


def is_liquid(row: dict, min_vol: int, min_oi: int) -> bool:
    vols = [row.get("call_volume", 0), row.get("put_volume", 0)]
    ois = [row.get("call_open_interest", 0), row.get("put_open_interest", 0)]
    return all(v >= min_vol for v in vols) and all(oi >= min_oi for oi in ois)


def build_spreads(
    ticker: str,
    expiry_dt: datetime,
    spot: Optional[float],
    merged: pl.DataFrame,
    config: Config,
    snapshot_time: datetime,
) -> pl.DataFrame:
    if merged.is_empty():
        return pl.DataFrame()
    valid_rows = [row for row in merged.iter_rows(named=True) if is_liquid(row, config.min_volume, config.min_open_interest)]
    if len(valid_rows) < 2:
        return pl.DataFrame()
    records: List[Dict] = []
    tte_years = max((expiry_dt - snapshot_time).total_seconds() / YEAR_SECONDS, 0.0001)
    for idx, left in enumerate(valid_rows):
        for right in valid_rows[idx + 1 :]:
            kl = float(left["strike"])
            ks = float(right["strike"])
            width = ks - kl
            if width < config.min_strike_gap or width > config.max_strike_gap:
                continue
            prices_mid = {
                "call_kl": float(left["call_mid"]),
                "call_ks": float(right["call_mid"]),
                "put_kl": float(left["put_mid"]),
                "put_ks": float(right["put_mid"]),
            }
            prices_bid = {
                "call_kl": float(left["call_bid"]),
                "call_ks": float(right["call_bid"]),
                "put_kl": float(left["put_bid"]),
                "put_ks": float(right["put_bid"]),
            }
            prices_ask = {
                "call_kl": float(left["call_ask"]),
                "call_ks": float(right["call_ask"]),
                "put_kl": float(left["put_ask"]),
                "put_ks": float(right["put_ask"]),
            }
            spread = BoxSpread(ticker, expiry_dt, kl, ks, prices_mid, tte_years)
            payoff = abs(width)
            mid_cost = spread.net_cost
            bid_cost = BoxSpread.calc_cost(**prices_bid)
            ask_cost = BoxSpread.calc_cost(**prices_ask)
            mid_rate = spread.implied_rate
            bid_rate = BoxSpread.calc_implied_rate(payoff, bid_cost, tte_years)
            ask_rate = BoxSpread.calc_implied_rate(payoff, ask_cost, tte_years)
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
                "bid_cost": bid_cost,
                "ask_cost": ask_cost,
                "mid_rate": mid_rate,
                "bid_rate": bid_rate,
                "ask_rate": ask_rate,
                "lend_borrow": spread.lend_borrow.name,
                "spot_price": spot or np.nan,
                "moneyness_kl": (kl / spot) if spot else np.nan,
                "moneyness_ks": (ks / spot) if spot else np.nan,
                "moneyness_mid": ((kl + ks) / 2.0 / spot) if spot else np.nan,
                "call_kl_volume": int(left["call_volume"]),
                "call_ks_volume": int(right["call_volume"]),
                "put_kl_volume": int(left["put_volume"]),
                "put_ks_volume": int(right["put_volume"]),
                "call_kl_open_interest": int(left["call_open_interest"]),
                "call_ks_open_interest": int(right["call_open_interest"]),
                "put_kl_open_interest": int(left["put_open_interest"]),
                "put_ks_open_interest": int(right["put_open_interest"]),
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


class SnapshotCollector:
    def __init__(self, client: TastytradeClient, config: Config) -> None:
        self.client = client
        self.config = config
        self.storage_dir = (Path(__file__).parent / config.snapshot_storage).resolve()
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def run_once(self) -> Optional[Path]:
        snapshot_time = datetime.now(timezone.utc)
        frames: List[pl.DataFrame] = []
        for ticker in self.config.tickers:
            try:
                expiries = self.client.get_expiries(ticker)
            except Exception as exc:  # pragma: no cover - network
                LOGGER.error("Failed to get expiries for %s: %s", ticker, exc)
                continue
            expiries = [e for e in expiries if e]
            expiries = expiries[: self.config.expiries_per_ticker]
            if not expiries:
                continue
            spot = self.client.get_underlying_price(ticker)
            for expiry in expiries:
                expiry_dt = parse_expiry(expiry)
                try:
                    quotes = self.client.get_option_quotes(ticker, expiry)
                except Exception as exc:  # pragma: no cover
                    LOGGER.error("Failed quotes %s %s: %s", ticker, expiry, exc)
                    continue
                merged = pivot_calls_puts(quotes_to_frame(quotes))
                spreads = build_spreads(ticker, expiry_dt, spot, merged, self.config, snapshot_time)
                if spreads.is_empty():
                    continue
                frames.append(spreads)
        if not frames:
            LOGGER.warning("No spreads generated in this snapshot")
            return None
        snapshot_df = pl.concat(frames, how="vertical")
        path = self._write_snapshot(snapshot_df, snapshot_time)
        self._prune_old_snapshots()
        LOGGER.info("Snapshot saved to %s (%d rows)", path.name, snapshot_df.height)
        return path

    def loop(self) -> None:
        interval = max(self.config.update_interval_seconds, 5)
        while True:
            start = time.time()
            try:
                self.run_once()
            except Exception as exc:  # pragma: no cover
                LOGGER.exception("Snapshot iteration failed: %s", exc)
            elapsed = time.time() - start
            sleep_for = max(interval - elapsed, 0)
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _write_snapshot(self, df: pl.DataFrame, ts: datetime) -> Path:
        timestamp = ts.strftime("%Y%m%d_%H%M%S")
        parquet_path = self.storage_dir / f"snapshot_{timestamp}.parquet"
        try:
            df.write_parquet(parquet_path)
            return parquet_path
        except Exception:  # pragma: no cover - optional dependency missing
            csv_path = parquet_path.with_suffix(".csv")
            df.write_csv(csv_path)
            return csv_path

    def _prune_old_snapshots(self) -> None:
        files = sorted(self.storage_dir.glob("snapshot_*"), key=lambda p: p.stat().st_mtime)
        excess = len(files) - self.config.max_snapshots_retained
        for path in files[:excess]:
            try:
                path.unlink()
            except OSError:  # pragma: no cover
                continue


def build_client() -> TastytradeClient:
    username, password = ensure_env_credentials()
    return TastytradeClient(username, password)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect live box spread snapshots")
    parser.add_argument("--loop", action="store_true", help="Continuously collect snapshots")
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to config.yaml (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = load_config(args.config)
    client = build_client()
    collector = SnapshotCollector(client, config)
    if args.loop:
        collector.loop()
    else:
        collector.run_once()


if __name__ == "__main__":
    main()
