from __future__ import annotations

import pandas as pd
import pytest

from src.forward_economic_assessment import capital_reserving_metrics, select_capacity


def test_capacity_prevents_overlaps_and_enforces_daily_limit() -> None:
    trades = pd.DataFrame([
        {"entry_date": "2026-09-02", "exit_date": "2026-09-10", "candidate_rank": 1, "symbol": "AAA"},
        {"entry_date": "2026-09-02", "exit_date": "2026-09-04", "candidate_rank": 2, "symbol": "BBB"},
        {"entry_date": "2026-09-03", "exit_date": "2026-09-11", "candidate_rank": 1, "symbol": "AAA"},
    ])
    accepted, rejected = select_capacity(trades, max_daily_entries=1, max_concurrent_positions=2)
    assert accepted["symbol"].tolist() == ["AAA"]
    assert set(rejected["rejection_reason"]) == {"daily_entry_cap", "symbol_overlap"}


def test_capital_reserving_metrics_marks_open_short_to_market() -> None:
    trades = pd.DataFrame([{
        "observation_id": "one", "signal_date": "2026-09-01", "candidate_rank": 1,
        "candidate_key": "locked", "symbol": "AAA", "direction": "short",
        "entry_date": "2026-09-02", "exit_date": "2026-09-04",
        "entry_touch_price": 10.0, "exit_touch_price": 9.0,
        "quote_gross_return": 0.10, "quote_net_return": 0.09,
    }])
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-02", "2026-09-03", "2026-09-04"]),
        "symbol": ["AAA", "AAA", "AAA"], "close": [12.0, 11.0, 9.0],
    })
    accepted, curve, metrics = capital_reserving_metrics(trades, prices, {
        "initial_capital": 100_000, "position_fraction": 0.05,
        "max_daily_entries": 3, "max_concurrent_positions": 10,
        "target_cagr": 0.40, "maximum_acceptable_drawdown": 0.25,
    })
    assert len(accepted) == 1
    assert curve.iloc[0]["equity"] == pytest.approx(98_950.0)
    assert curve.iloc[-1]["equity"] == pytest.approx(100_450.0)
    assert metrics["economic_mtm_coverage"] == 1.0
    assert metrics["economic_max_drawdown"] == pytest.approx(0.0105)
    assert metrics["economic_gate_passed"] is True


def test_missing_marks_fail_economic_gate() -> None:
    trades = pd.DataFrame([{
        "observation_id": "one", "signal_date": "2026-09-01", "candidate_rank": 1,
        "candidate_key": "locked", "symbol": "AAA", "direction": "short",
        "entry_date": "2026-09-02", "exit_date": "2026-09-04",
        "entry_touch_price": 10.0, "exit_touch_price": 9.0,
        "quote_gross_return": 0.10, "quote_net_return": 0.09,
    }])
    prices = pd.DataFrame({"date": pd.to_datetime(["2026-09-02"]), "symbol": ["BBB"], "close": [1.0]})
    _, _, metrics = capital_reserving_metrics(trades, prices, {})
    assert metrics["economic_mtm_coverage"] == 0.0
    assert metrics["economic_gate_passed"] is False


def test_short_borrow_is_accrued_daily_and_reduces_ending_equity() -> None:
    trades = pd.DataFrame([{
        "observation_id": "one", "signal_date": "2026-09-01", "candidate_rank": 1,
        "candidate_key": "locked", "symbol": "AAA", "direction": "short",
        "entry_date": "2026-09-02", "exit_date": "2026-09-04",
        "entry_touch_price": 10.0, "exit_touch_price": 9.0,
        "quote_gross_return": 0.10, "quote_net_return": 0.09,
    }])
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-02", "2026-09-03", "2026-09-04"]),
        "symbol": ["AAA", "AAA", "AAA"], "close": [10.0, 10.0, 9.0],
    })
    _, curve, metrics = capital_reserving_metrics(trades, prices, {
        "short_borrow_bps_annual": 36525,
    })
    assert curve["daily_borrow_cost"].sum() == pytest.approx(100.0)
    assert metrics["economic_ending_equity"] == pytest.approx(100_350.0)


def test_marked_gross_exposure_limit_blocks_gate() -> None:
    trades = pd.DataFrame([{
        "observation_id": "one", "signal_date": "2026-09-01", "candidate_rank": 1,
        "candidate_key": "locked", "symbol": "AAA", "direction": "short",
        "entry_date": "2026-09-02", "exit_date": "2026-09-04",
        "entry_touch_price": 10.0, "exit_touch_price": 9.0,
        "quote_gross_return": 0.10, "quote_net_return": 0.09,
    }])
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-02", "2026-09-03", "2026-09-04"]),
        "symbol": ["AAA", "AAA", "AAA"], "close": [30.0, 10.0, 9.0],
    })
    _, _, metrics = capital_reserving_metrics(trades, prices, {
        "position_fraction": 0.20, "maximum_gross_exposure": 0.50,
        "maximum_acceptable_drawdown": 1.0,
    })
    assert metrics["economic_max_gross_exposure"] > 0.50
    assert metrics["economic_gate_passed"] is False


def test_unaffordable_whole_share_is_rejected() -> None:
    trades = pd.DataFrame([{
        "observation_id": "one", "signal_date": "2026-09-01", "candidate_rank": 1,
        "candidate_key": "locked", "symbol": "AAA", "direction": "short",
        "entry_date": "2026-09-02", "exit_date": "2026-09-04",
        "entry_touch_price": 10.0, "exit_touch_price": 9.0,
        "quote_gross_return": 0.10, "quote_net_return": 0.09,
    }])
    prices = pd.DataFrame({"date": pd.to_datetime(["2026-09-02"]), "symbol": ["AAA"], "close": [10.0]})
    accepted, _, metrics = capital_reserving_metrics(
        trades, prices, {"initial_capital": 100, "position_fraction": 0.05}
    )
    assert accepted.empty
    assert metrics["economic_rejected_trades"] == 1
    assert metrics["economic_gate_passed"] is False
