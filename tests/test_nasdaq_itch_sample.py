from types import SimpleNamespace

import pandas as pd
import pytest

from src.nasdaq_itch_sample import quote_diagnostics, top_quote


def level(price, volumes):
    return SimpleNamespace(price=price, queue=[SimpleNamespace(volume=volume) for volume in volumes])


def test_top_quote_aggregates_touch_depth_and_scales_price():
    lob = SimpleNamespace(bid_levels=[level(100_000, [10, 20])], ask_levels=[level(100_100, [5, 15])])
    assert top_quote(lob) == (10.0, 10.01, 30.0, 20.0)


def test_top_quote_requires_two_sided_book():
    lob = SimpleNamespace(bid_levels=[level(100_000, [10])], ask_levels=[])
    assert top_quote(lob) is None


def test_quote_diagnostics_reports_spread_and_frequency():
    quotes = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02 14:30Z", "2024-01-02 15:30Z"]),
            "bid": [9.99, 9.99],
            "ask": [10.01, 10.01],
            "bid_size": [100, 200],
            "ask_size": [200, 300],
        }
    )
    result = quote_diagnostics(quotes)
    assert result["quotes"] == 2
    assert result["quote_changes_per_hour"] == 2
    assert result["median_spread_bps"] == pytest.approx(20)
