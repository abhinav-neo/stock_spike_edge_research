import numpy as np
import pandas as pd

from src.intraday_bar_research import add_causal_regime, normalize_bars, performance, strategy_returns


def bars(periods=240):
    timestamps = pd.date_range("2024-01-02 14:30Z", periods=periods, freq="5min")
    rows = []
    for symbol, slope in (("SPY", 0.01), ("AAA", 0.03), ("BBB", -0.01)):
        close = 100 + slope * np.arange(periods) + np.sin(np.arange(periods) / 7)
        for timestamp, value in zip(timestamps, close, strict=True):
            rows.append({"timestamp": timestamp, "symbol": symbol, "open": value, "high": value, "low": value, "close": value, "volume": 1000})
    return pd.DataFrame(rows)


def test_regime_is_invariant_to_future_prices():
    original = add_causal_regime(bars(), minimum_history=20)
    changed = bars()
    changed.loc[changed.timestamp >= changed.timestamp.unique()[180], "close"] *= 3
    revised = add_causal_regime(changed, minimum_history=20)
    cutoff = original.timestamp.unique()[180]
    assert original.loc[original.timestamp < cutoff, "regime"].tolist() == revised.loc[revised.timestamp < cutoff, "regime"].tolist()


def test_strategy_enters_on_next_bar_and_charges_cost():
    frame = add_causal_regime(bars(), minimum_history=20)
    candidate = {"mode": "reversal", "book": "long_only", "lookback": 1, "holding": 1, "regime": "all", "top_k": 1}
    free = strategy_returns(frame, candidate, round_trip_cost_bps=0)
    costly = strategy_returns(frame, candidate, round_trip_cost_bps=10)
    assert np.allclose(free.net_return - costly.net_return, 0.001)
    assert (free.exit_timestamp > free.timestamp).all()


def test_performance_reports_trade_frequency():
    returns = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02 15:00Z", "2024-01-02 16:00Z"]),
            "exit_timestamp": pd.to_datetime(["2024-01-02 15:05Z", "2024-01-02 16:05Z"]),
            "net_return": [0.01, -0.005],
            "trade_legs": [2, 2],
        }
    )
    result = performance(returns)
    assert result["trades_per_day"] == 4
    assert result["trade_legs"] == 4


def test_normalize_rejects_missing_bar_columns():
    try:
        normalize_bars(pd.DataFrame({"timestamp": []}))
    except ValueError as error:
        assert "missing required columns" in str(error)
    else:
        raise AssertionError("missing columns should fail")
