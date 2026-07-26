import numpy as np
import pandas as pd

from src.dependency_validation import (
    jaccard_similarity,
    monthly_block_bootstrap,
    simulate_portfolio,
)


def test_jaccard_similarity_handles_overlap():
    left = pd.Series([True, True, False, False])
    right = pd.Series([True, False, True, False])
    assert jaccard_similarity(left, right) == 1 / 3


def test_monthly_block_bootstrap_preserves_positive_edge():
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    trades = pd.DataFrame({"event_date": dates, "net_return": np.full(len(dates), 0.10)})
    result = monthly_block_bootstrap(trades, samples=200, seed=7)
    assert result["ci_low"] > 0
    assert result["probability_positive"] == 1.0


def test_portfolio_respects_concurrent_position_limit():
    trades = pd.DataFrame({
        "event_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        "exit_date": pd.to_datetime(["2020-02-01", "2020-02-02", "2020-02-03"]),
        "net_return": [0.10, 0.10, 0.10],
    })
    ledger, summary = simulate_portfolio(
        trades,
        {
            "initial_capital": 100000,
            "position_fraction": 0.10,
            "max_concurrent_positions": 2,
            "max_daily_entries": 1,
            "stop_loss": None,
        },
    )
    assert len(ledger) == 2
    assert summary["trades"] == 2
    assert summary["ending_capital"] == 102000


def test_stop_loss_caps_trade_loss():
    trades = pd.DataFrame({
        "event_date": pd.to_datetime(["2020-01-01"]),
        "exit_date": pd.to_datetime(["2020-02-01"]),
        "net_return": [-0.80],
    })
    ledger, _ = simulate_portfolio(
        trades,
        {
            "initial_capital": 100000,
            "position_fraction": 0.10,
            "max_concurrent_positions": 1,
            "max_daily_entries": 1,
            "stop_loss": 0.25,
        },
    )
    assert ledger.iloc[0]["return"] == -0.25
