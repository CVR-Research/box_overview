from typing import Any, Dict, List

from live_box_spreads.providers.alpaca_rest import AlpacaRestProvider


def test_get_option_chain_calls_snapshots(monkeypatch):
    provider = AlpacaRestProvider("key", "secret")
    calls: List[Dict[str, Any]] = []

    def fake_request(self, method, path, *, params=None):
        calls.append({"path": path, "params": params})
        return {
            "snapshots": {
                "AAPL260118C00100000": {
                    "symbol": "AAPL260118C00100000",
                    "expiration_date": "2026-01-18",
                    "strike": 100.0,
                    "option_type": "call",
                }
            }
        }

    monkeypatch.setattr(AlpacaRestProvider, "_request", fake_request)
    items = provider.get_option_chain("AAPL")
    assert items
    assert calls[0]["path"] == "/v1beta1/options/snapshots/AAPL"


def test_extract_expiries_from_chain():
    provider = AlpacaRestProvider("key", "secret")
    items = [
        {"symbol": "AAPL260118C00100000"},
        {"symbol": "AAPL260215P00100000"},
    ]
    expiries = provider.extract_expiries(items)
    assert sorted(expiries) == ["2026-01-18", "2026-02-15"]


def test_quotes_for_expiry_filters():
    provider = AlpacaRestProvider("key", "secret")
    items = [
        {
            "symbol": "AAPL260118C00100000",
            "expiration_date": "2026-01-18",
            "strike": 100.0,
            "option_type": "call",
            "bid": 1.0, "ask": 1.2, "last": 1.1, "mark": 1.1,
            "volume": 10, "open_interest": 50,
        },
        {
            "symbol": "AAPL260215P00100000",
            "expiration_date": "2026-02-15",
            "strike": 100.0,
            "option_type": "put",
            "bid": 1.3, "ask": 1.5, "last": 1.4, "mark": 1.4,
            "volume": 12, "open_interest": 55,
        },
    ]
    quotes = provider.quotes_for_expiry("AAPL", items, "2026-01-18")
    assert {q.symbol for q in quotes} == {"AAPL260118C00100000"}


def test_get_option_chain_returns_empty_on_no_snapshots(monkeypatch):
    provider = AlpacaRestProvider("key", "secret")

    def fake_request(self, method, path, *, params=None):
        return {"snapshots": {}}

    monkeypatch.setattr(AlpacaRestProvider, "_request", fake_request)
    result = provider.get_option_chain("AAPL")
    assert result == []
