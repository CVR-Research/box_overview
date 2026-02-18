from datetime import datetime, timedelta

import pandas as pd

from boxspread import BoxSpread


def test_calc_cost_and_rate_sign():
    cost = BoxSpread.calc_cost(5.0, 1.0, 1.0, 5.0)
    assert cost == 8.0
    rate = BoxSpread.calc_implied_rate(10.0, cost, 1.0)
    assert rate > 0


def test_batch_from_snapshot_uses_all_rows():
    frame = pd.DataFrame(
        {
            "ticker": ["SPX", "SPX"],
            "expiry": [datetime(2026, 1, 1), datetime(2026, 1, 1)],
            "kl": [4000.0, 4010.0],
            "ks": [4050.0, 4060.0],
            "call_kl": [10.0, 11.0],
            "call_ks": [9.0, 10.0],
            "put_kl": [8.0, 9.0],
            "put_ks": [7.0, 8.0],
        }
    )
    spreads = BoxSpread.batch_from_snapshot(frame, tte=0.5)
    assert len(spreads) == 2
    assert spreads[0].ticker == "SPX"
