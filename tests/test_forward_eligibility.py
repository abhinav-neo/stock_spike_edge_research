from __future__ import annotations

import pandas as pd

from src.forward_eligibility import evaluate_eligibility


def test_ineligible_short_is_quarantined_with_reason() -> None:
    ledger = pd.DataFrame([{
        "signal_date": "2026-08-10", "candidate_key": "locked", "symbol": "AAA", "direction": "short"
    }])
    snapshots = pd.DataFrame([{
        "snapshot_date": "2026-08-11", "signal_date": "2026-08-10", "symbol": "AAA", "direction": "short",
        "tradable": True, "shortable": False, "easy_to_borrow": False,
    }])
    result = evaluate_eligibility(ledger, snapshots)
    assert not bool(result.iloc[0]["broker_eligible"])
    assert result.iloc[0]["rejection_reason"] == "not_shortable"


def test_eligible_short_requires_tradable_shortable_and_easy_to_borrow() -> None:
    ledger = pd.DataFrame([{
        "signal_date": "2026-08-10", "candidate_key": "locked", "symbol": "AAA", "direction": "short"
    }])
    snapshots = pd.DataFrame([{
        "snapshot_date": "2026-08-11", "signal_date": "2026-08-10", "symbol": "AAA", "direction": "short",
        "tradable": True, "shortable": True, "easy_to_borrow": True,
    }])
    result = evaluate_eligibility(ledger, snapshots)
    assert bool(result.iloc[0]["broker_eligible"])
    assert result.iloc[0]["rejection_reason"] == ""


def test_eligibility_is_frozen_to_first_broker_snapshot() -> None:
    ledger = pd.DataFrame([{
        "signal_date": "2026-08-10", "candidate_key": "locked", "symbol": "AAA", "direction": "short"
    }])
    snapshots = pd.DataFrame([
        {"snapshot_date": "2026-08-10", "signal_date": "2026-08-10", "symbol": "AAA", "direction": "short",
         "tradable": True, "shortable": True, "easy_to_borrow": True},
        {"snapshot_date": "2026-08-11", "signal_date": "2026-08-10", "symbol": "AAA", "direction": "short",
         "tradable": True, "shortable": False, "easy_to_borrow": False},
    ])
    result = evaluate_eligibility(ledger, snapshots)
    assert bool(result.iloc[0]["broker_eligible"])
