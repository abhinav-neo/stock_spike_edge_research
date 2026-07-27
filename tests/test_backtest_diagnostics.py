from __future__ import annotations

import pandas as pd
import pytest

from src.backtest_diagnostics import (
    benchmark_comparison,
    cost_stress_test,
    grouped_performance,
    worst_trade_diagnostics,
)


def test_grouped_performance_calculates_profit_factor() -> None:
    trades = pd.DataFrame({
        "family": ["momentum", "momentum", "gap_fade"],
        "net_return": [0.20, -0.10, 0.05],
    })
    result = grouped_performance(trades, ["family"])
    momentum = result[result["family"].eq("momentum")].iloc[0]
    assert momentum["trades"] == 2
    assert momentum["win_rate"] == pytest.approx(0.5)
    assert momentum["profit_factor"] == pytest.approx(2.0)


def test_cost_stress_test_reduces_expectancy() -> None:
    trades = pd.DataFrame({"net_return": [0.02, 0.01, -0.01]})
    result = cost_stress_test(trades, [0, 100])
    assert result.loc[1, "mean_net_return"] < result.loc[0, "mean_net_return"]
    assert result.loc[1, "mean_net_return"] == pytest.approx(result.loc[0, "mean_net_return"] - 0.01)


def test_benchmark_comparison_builds_strategy_and_benchmark_rows() -> None:
    dates = pd.date_range("2024-01-02", periods=4, freq="B")
    equity = pd.DataFrame({"date": dates, "equity": [100000, 101000, 99000, 105000]})
    prices = pd.DataFrame({
        "symbol": ["SPY"] * 4,
        "date": dates,
        "close": [400, 404, 402, 420],
    })
    result = benchmark_comparison(equity, prices, "SPY")
    assert result["series"].tolist() == ["strategy", "SPY"]
    assert result.loc[0, "total_return"] == pytest.approx(0.05)
    assert result.loc[1, "total_return"] == pytest.approx(0.05)


def test_worst_trade_diagnostics_orders_losses_first() -> None:
    trades = pd.DataFrame({
        "symbol": ["AAA", "BBB", "CCC"],
        "net_return": [0.1, -0.4, -0.2],
    })
    result = worst_trade_diagnostics(trades, limit=2)
    assert result["symbol"].tolist() == ["BBB", "CCC"]
