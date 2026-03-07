from live_box_spreads.core.opra import extract_underlying, parse_opra_symbol


def test_parse_call():
    result = parse_opra_symbol("SPX260118C04000000")
    assert result is not None
    root, expiry, strike, opt_type = result
    assert root == "SPX"
    assert expiry == "2026-01-18"
    assert strike == 4000.0
    assert opt_type == "call"


def test_parse_put():
    result = parse_opra_symbol("AAPL260215P00100000")
    assert result is not None
    assert result[0] == "AAPL"
    assert result[1] == "2026-02-15"
    assert result[2] == 100.0
    assert result[3] == "put"


def test_parse_invalid():
    assert parse_opra_symbol("INVALID") is None
    assert parse_opra_symbol("") is None


def test_extract_underlying():
    assert extract_underlying("SPX260118C04000000") == "SPX"
    assert extract_underlying("UNKNOWN") == "UNKNOWN"
