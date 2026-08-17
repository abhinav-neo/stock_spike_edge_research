from datetime import datetime, timezone

import pandas as pd

from src.forward_event_risk import append_first_capture, build_evidence


def test_build_evidence_marks_capture_before_entry() -> None:
    ledger = pd.DataFrame(
        [{
            "signal_date": "2026-08-17", "entry_date": "2026-08-18", "symbol": "ABC",
            "direction": "short", "candidate_key": "locked",
        }]
    )
    actions = pd.DataFrame(
        [{"symbol": "ABC", "action_type": "reverse_splits", "effective_date": pd.Timestamp("2026-08-01")}]
    )
    halts = pd.DataFrame([{"symbol": "ABC", "halt_date": pd.Timestamp("2026-08-17")}])
    result = build_evidence(ledger, actions, halts, datetime(2026, 8, 17, 23, tzinfo=timezone.utc))
    assert result.loc[0, "captured_before_entry"]
    assert result.loc[0, "event_day_halt"]
    assert result.loc[0, "reverse_split_within_30d"]


def test_append_first_capture_is_immutable() -> None:
    old = pd.DataFrame([{"observation_id": "x", "captured_at_utc": "old"}])
    new = pd.DataFrame([{"observation_id": "x", "captured_at_utc": "new"}])
    for column in [
        "signal_date", "symbol", "captured_before_entry", "event_day_halt",
        "reverse_split_within_30d", "reverse_split_within_90d",
        "reverse_split_within_180d", "reverse_split_within_365d",
    ]:
        old[column] = ""
        new[column] = ""
    result = append_first_capture(old, new)
    assert result.loc[0, "captured_at_utc"] == "old"
