from pathlib import Path

import pytest
import yaml

from live_box_spreads.config import Config, load_credentials


def test_from_dict_defaults():
    config = Config.from_dict({})
    assert config.tickers == ["SPX"]
    assert config.expiries_per_ticker == 2
    assert config.min_volume == 0


def test_from_dict_custom():
    config = Config.from_dict({
        "tickers": ["VIX", "DJX"],
        "min_volume": 50,
        "max_strike_gap": 200.0,
    })
    assert config.tickers == ["VIX", "DJX"]
    assert config.min_volume == 50
    assert config.max_strike_gap == 200.0


def test_from_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump({"tickers": ["SPX"], "min_volume": 25}))
    config = Config.from_yaml(config_path)
    assert config.tickers == ["SPX"]
    assert config.min_volume == 25


def test_frozen():
    config = Config.from_dict({})
    with pytest.raises(AttributeError):
        config.min_volume = 999


def test_load_credentials_missing(monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_KEY_ID", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="ALPACA_API_KEY"):
        load_credentials()


def test_load_credentials_ok(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_API_SECRET", "test_secret")
    key, secret = load_credentials()
    assert key == "test_key"
    assert secret == "test_secret"
