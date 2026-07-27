from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest_performance_report import calculate_performance_metrics, period_returns


def test_calculate_performance_metrics_reports_trade_and_risk_statistics() -> None:
    trades = pd.DataFrame([
        {"entry_date": "2024-01-02", "exit_date": "2024-01-05", "net_return": 0.10},
        {"entry_date": "2024-02-01", "exit_date": "2024-02-02", "net_return": -0.05},
        {"entry_date": "2024-03-01", "exit_date": "2024-03-08", "net_return": 0.02},
    ])
    dates = pd.bdate_range("2024-01-02", "2024-12-31")
    equity_values = np.linspace(100000.0, 110000.0, len(dates))
    equity = pd.DataFrame({"date": dates, "equity": equity_values})
    equity["daily_return"] = equity["equity"].pct_change().fillna(0.0)
    equity["drawdown"] = equity["equity"] / equity["equity"].cummax() - 1.0

    metrics = calculate_performance_metrics(trades, equity, initial_capital=100000.0)

    assert metrics["trades"] == 3
    assert metrics["winning_trades"] == 2
    assert metrics["losing_trades"] == 1
    assert metrics["win_rate"] == pytest.approx(2 / 3)
    assert metrics["profit_factor"] == pytest.approx(2.4)
    assert metrics["expectancy"] == pytest.approx((0.10 - 0.05 + 0.02) / 3)
    assert metrics["final_equity"] == pytest.approx(110000.0)
    assert metrics["total_return"] == pytest.approx(0.10)
    assert metrics["cagr"] > 0
    assert metrics["max_drawdown"] == pytest.approx(0.0)


def test_empty_report_keeps_cagr_unavailable() -> None:
    metrics = calculate_performance_metrics(pd.DataFrame(), pd.DataFrame(), 100000.0)
    assert metrics["trades"] == 0
    assert np.isnan(metrics["cagr"])
    assert metrics["final_equity"] == pytest.approx(100000.0)


def test_period_returns_produces_calendar_rows() -> None:
    equity = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-31", "2024-02-29"]),
        "equity": [100000.0, 105000.0, 110000.0],
    })
    monthly = period_returns(equity, "ME")
    assert len(monthly) == 2
    assert monthly.iloc[0]["return"] == pytest.approx(0.05)
    assert monthly.iloc[1]["return"] == pytest.approx(110000.0 / 105000.0 - 1.0)
