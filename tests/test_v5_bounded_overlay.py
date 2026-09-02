import pandas as pd
import pytest

from src.v5_bounded_overlay import combined_metrics, directional_paths


def test_directional_paths_respect_long_and_short_signs():
    prices = pd.DataFrame({"symbol": ["A"] * 5, "date": pd.date_range("2024-01-02", periods=5, freq="B"), "open": [10] * 5, "close": [10, 11, 12, 11, 12]})
    base = {"symbol": ["A"], "event_date": [pd.Timestamp("2024-01-01")]}
    long_paths, long_done = directional_paths(pd.DataFrame({**base, "side": ["long"]}), prices, 0, 0, "l")
    short_paths, short_done = directional_paths(pd.DataFrame({**base, "side": ["short"]}), prices, 0, 0, "s")
    assert long_done.loc[0, "net_return"] == pytest.approx(0.20)
    assert short_done.loc[0, "net_return"] == pytest.approx(-0.20)
    assert long_paths.loc[4, "mark_return"] == pytest.approx(0.20)


def test_combined_portfolio_reserves_gross_exposure():
    dates = pd.date_range("2024-01-02", periods=2, freq="B")
    paths = pd.DataFrame({"trade_id": ["a", "a"], "date": dates, "mark_return": [0.0, 0.1]})
    trades = pd.DataFrame({"trade_id": ["a"], "exit_date": [dates[-1]], "net_return": [0.1]})
    spy = pd.DataFrame({"date": dates, "benchmark": [100.0, 110.0]})
    metrics, _ = combined_metrics(paths, trades, spy, 0.1)
    assert metrics["core_weight"] == pytest.approx(0.9)
    assert metrics["maximum_gross_exposure"] == pytest.approx(1.0)
