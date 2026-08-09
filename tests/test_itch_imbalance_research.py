import numpy as np
import pandas as pd

from src.itch_imbalance_research import build_imbalance_signals, online_state_persistence


def quotes():
    timestamps = pd.date_range("2024-01-02 14:30Z", periods=20, freq="s")
    return pd.DataFrame(
        {
            "symbol": ["AAA"] * 20,
            "timestamp": timestamps,
            "bid": [10.00] * 20,
            "ask": [10.01] * 20,
            "bid_size": [900] * 20,
            "ask_size": [100] * 20,
        }
    )


def test_online_persistence_is_invariant_to_future_states():
    original = online_state_persistence(pd.Series([1, 2, 2, 1, 0, 0]))
    changed = online_state_persistence(pd.Series([1, 2, 2, 2, 2, 2]))
    assert np.allclose(original[:3], changed[:3])


def test_signals_obey_holding_cooldown_and_capital():
    signals = build_imbalance_signals(quotes(), 0.5, 0.0, holding_seconds=5)
    assert len(signals) == 4
    assert signals["side"].eq("buy").all()
    assert signals["quantity"].eq(99).all()
    assert signals["signal_timestamp"].diff().dropna().ge(pd.Timedelta(seconds=5)).all()
