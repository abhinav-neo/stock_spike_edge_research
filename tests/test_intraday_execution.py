import pandas as pd
import pytest

from src.intraday_execution import ExecutionCosts, account_metrics, simulate_round_trips, validate_quotes


def quotes():
    return pd.DataFrame(
        {
            "symbol": ["AAA"] * 4,
            "timestamp": pd.to_datetime(
                ["2024-01-02 14:30:00.000Z", "2024-01-02 14:30:00.100Z", "2024-01-02 14:31:00.000Z", "2024-01-02 14:31:00.100Z"]
            ),
            "bid": [9.99, 10.00, 10.09, 10.10],
            "ask": [10.01, 10.02, 10.11, 10.12],
            "bid_size": [1000] * 4,
            "ask_size": [1000] * 4,
        }
    )


def signal(quantity=10):
    return pd.DataFrame(
        {
            "symbol": ["AAA"],
            "signal_timestamp": ["2024-01-02 14:30:00.000Z"],
            "exit_timestamp": ["2024-01-02 14:31:00.000Z"],
            "side": ["buy"],
            "quantity": [quantity],
        }
    )


def test_latency_prevents_same_timestamp_fill():
    fills, _ = simulate_round_trips(quotes(), signal(), ExecutionCosts(latency_ms=100, impact_bps_at_full_touch=0))
    assert fills.loc[0, "entry_timestamp"] == pd.Timestamp("2024-01-02 14:30:00.100Z")
    assert fills.loc[0, "entry_price"] == pytest.approx(10.02)
    assert fills.loc[0, "exit_price"] == pytest.approx(10.10)


def test_spread_impact_and_commission_reduce_pnl():
    zero, _ = simulate_round_trips(quotes(), signal(), ExecutionCosts(latency_ms=100, impact_bps_at_full_touch=0))
    costly, _ = simulate_round_trips(
        quotes(), signal(), ExecutionCosts(latency_ms=100, impact_bps_at_full_touch=10, commission_per_share=0.01)
    )
    assert costly.loc[0, "net_pnl"] < zero.loc[0, "net_pnl"]


def test_touch_participation_rejects_oversized_order():
    fills, rejected = simulate_round_trips(quotes(), signal(300), ExecutionCosts(maximum_participation=0.25))
    assert fills.empty
    assert rejected.loc[0, "reason"] == "touch_participation_limit"


def test_crossed_quotes_are_rejected():
    frame = quotes()
    frame.loc[0, "bid"] = 11
    with pytest.raises(ValueError, match="crossed"):
        validate_quotes(frame)


def test_account_metrics_use_ten_thousand_default():
    fills, _ = simulate_round_trips(quotes(), signal(), ExecutionCosts(latency_ms=100, impact_bps_at_full_touch=0))
    equity, summary = account_metrics(fills)
    assert summary["initial_capital"] == 10_000
    assert summary["ending_equity"] == pytest.approx(10_000.8)
    assert equity.loc[0, "equity"] == pytest.approx(10_000.8)
