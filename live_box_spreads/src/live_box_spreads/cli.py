"""CLI entry points for the Live Box Spread Monitor."""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


def _load_env(env_path: Path | None = None) -> None:
    """Load .env file if it exists."""
    candidates = [
        env_path,
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent.parent / ".env",
    ]
    for path in candidates:
        if path and path.exists():
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    os.environ.setdefault(key, value)
            return


def main_ingest() -> None:
    """Entry point: snapshot collection."""
    parser = argparse.ArgumentParser(description="Collect live box spread snapshots")
    parser.add_argument("--loop", action="store_true", help="Continuously collect snapshots")
    parser.add_argument("--stream", action="store_true", help="Use WebSocket streaming")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _load_env()

    from live_box_spreads.config import Config, find_config, load_credentials
    from live_box_spreads.providers.alpaca_rest import AlpacaRestProvider

    config_path = args.config or find_config()
    config = Config.from_yaml(config_path)
    api_key, api_secret = load_credentials()
    provider = AlpacaRestProvider(api_key, api_secret)
    storage_dir = (config_path.parent / config.snapshot_storage).resolve()

    if args.stream:
        from live_box_spreads.sources.stream_source import StreamSource

        source = StreamSource(provider, config, api_key, api_secret)
        source.start()
        # In stream mode with --loop, periodically write snapshots from stream
        if args.loop:
            import time
            from live_box_spreads.sources.rest_source import RestSource

            rest = RestSource(provider, config, storage_dir=storage_dir)
            interval = max(config.update_interval_seconds, 5)
            while True:
                start = time.time()
                try:
                    rest.run_once()
                except Exception as exc:
                    logging.getLogger(__name__).exception("Snapshot iteration failed: %s", exc)
                elapsed = time.time() - start
                sleep_for = max(interval - elapsed, 0)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        else:
            from live_box_spreads.sources.rest_source import RestSource

            rest = RestSource(provider, config, storage_dir=storage_dir)
            rest.run_once()
    else:
        from live_box_spreads.sources.rest_source import RestSource

        source = RestSource(provider, config, storage_dir=storage_dir)
        if args.loop:
            source.loop()
        else:
            source.run_once()


def main_dashboard() -> None:
    """Entry point: dashboard launch."""
    parser = argparse.ArgumentParser(description="Launch box spread dashboard")
    parser.add_argument("--stream", action="store_true", help="Use WebSocket streaming")
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    parser.add_argument("--port", type=int, default=8050, help="Dashboard port")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _load_env()

    from live_box_spreads.config import Config, find_config, load_credentials
    from live_box_spreads.dashboard.app import create_app
    from live_box_spreads.providers.alpaca_rest import AlpacaRestProvider

    config_path = args.config or find_config()
    config = Config.from_yaml(config_path)
    storage_dir = (config_path.parent / config.snapshot_storage).resolve()

    if args.stream:
        api_key, api_secret = load_credentials()
        provider = AlpacaRestProvider(api_key, api_secret)
        from live_box_spreads.sources.stream_source import StreamSource

        source = StreamSource(provider, config, api_key, api_secret)
        update_ms = 2000
    else:
        from live_box_spreads.sources.snapshot_source import SnapshotSource

        source = SnapshotSource(storage_dir, history_minutes=config.surface_history_minutes)
        update_ms = config.update_interval_seconds * 1000

    app = create_app(source, config, update_interval_ms=update_ms)
    app.run(debug=args.debug, port=args.port, use_reloader=False, use_debugger=False)


def main() -> None:
    """Unified entry point with subcommands."""
    parser = argparse.ArgumentParser(description="Live Box Spread Monitor")
    sub = parser.add_subparsers(dest="command")

    ingest_p = sub.add_parser("ingest", help="Collect snapshots")
    ingest_p.add_argument("--loop", action="store_true")
    ingest_p.add_argument("--stream", action="store_true")
    ingest_p.add_argument("--config", type=Path, default=None)

    dash_p = sub.add_parser("dashboard", help="Launch dashboard")
    dash_p.add_argument("--stream", action="store_true")
    dash_p.add_argument("--config", type=Path, default=None)
    dash_p.add_argument("--debug", action="store_true")
    dash_p.add_argument("--port", type=int, default=8050)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _load_env()

    from live_box_spreads.config import Config, find_config, load_credentials
    from live_box_spreads.providers.alpaca_rest import AlpacaRestProvider

    config_path = args.config or find_config()
    config = Config.from_yaml(config_path)
    storage_dir = (config_path.parent / config.snapshot_storage).resolve()

    if args.command == "ingest":
        api_key, api_secret = load_credentials()
        provider = AlpacaRestProvider(api_key, api_secret)
        if args.stream:
            from live_box_spreads.sources.stream_source import StreamSource

            stream = StreamSource(provider, config, api_key, api_secret)
            stream.start()
        from live_box_spreads.sources.rest_source import RestSource

        rest = RestSource(provider, config, storage_dir=storage_dir)
        if args.loop:
            rest.loop()
        else:
            rest.run_once()

    elif args.command == "dashboard":
        if args.stream:
            api_key, api_secret = load_credentials()
            provider = AlpacaRestProvider(api_key, api_secret)
            from live_box_spreads.sources.stream_source import StreamSource

            source = StreamSource(provider, config, api_key, api_secret)
            update_ms = 2000
        else:
            from live_box_spreads.sources.snapshot_source import SnapshotSource

            source = SnapshotSource(
                storage_dir, history_minutes=config.surface_history_minutes
            )
            update_ms = config.update_interval_seconds * 1000
        from live_box_spreads.dashboard.app import create_app

        app = create_app(source, config, update_interval_ms=update_ms)
        app.run(
            debug=args.debug, port=args.port,
            use_reloader=False, use_debugger=False,
        )


if __name__ == "__main__":
    main()
