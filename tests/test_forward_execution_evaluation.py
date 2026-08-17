from __future__ import annotations

import pandas as pd
import pytest

from src.forward_execution_evaluation import evaluate_executions, touch_fill
from src.forward_quote_capture import observation_id


def quotes() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["AAA", "AAA"],
        "timestamp": pd.to_datetime(["2026-08-11T13:30:00Z", "2026-08-11T13:34:59Z"]),
        "bid": [10.00, 9.90], "ask": [10.02, 9.92], "bid_size": [100, 100], "ask_size": [100, 100],
    })


def test_touch_fill_uses_adverse_side_for_long_and_short() -> None:
    _, long_entry, _ = touch_fill(quotes(), "long", "entry")
    _, short_entry, _ = touch_fill(quotes(), "short", "entry")
    _, long_exit, _ = touch_fill(quotes(), "long", "exit")
    _, short_exit, _ = touch_fill(quotes(), "short", "exit")
    assert (long_entry, short_entry, long_exit, short_exit) == (10.02, 10.00, 9.90, 9.92)


def test_execution_evaluation_reconciles_quote_and_bar_returns(tmp_path) -> None:
    ledger = pd.DataFrame([{
        "signal_date": "2026-08-10", "candidate_key": "locked", "symbol": "AAA", "direction": "short",
        "observation_status": "SETTLED", "net_return": 0.05,
    }])
    base = tmp_path / f"observation={observation_id(ledger.iloc[0])}"
    entry = quotes()
    exit_quotes = quotes().assign(
        timestamp=pd.to_datetime(["2026-08-17T19:55:00Z", "2026-08-17T19:59:59Z"]),
        bid=[9.00, 8.98], ask=[9.02, 9.00],
    )
    for phase, frame in (("entry", entry), ("exit", exit_quotes)):
        path = base / f"phase={phase}" / "quotes.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    result = evaluate_executions(ledger, tmp_path, 100)
    assert len(result) == 1
    assert result.iloc[0]["quote_net_return"] == pytest.approx(0.09)
    assert result.iloc[0]["execution_delta"] == pytest.approx(0.04)
