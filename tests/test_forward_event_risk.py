from datetime import datetime, timezone

import pandas as pd

from src.forward_event_risk import append_first_capture, build_evidence, normalize_capture_flags


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


def test_missing_entry_date_uses_next_weekday_open() -> None:
    ledger = pd.DataFrame(
        [{
            "signal_date": "2026-08-17", "entry_date": pd.NA, "symbol": "ABC",
            "direction": "short", "candidate_key": "locked",
        }]
    )
    result = build_evidence(
        ledger, pd.DataFrame(columns=["action_type", "symbol", "effective_date"]),
        pd.DataFrame(), datetime(2026, 8, 18, 3, 1, tzinfo=timezone.utc),
    )
    assert result.loc[0, "captured_before_entry"]


def test_capture_flag_migration_preserves_timestamp() -> None:
    frame = pd.DataFrame([{
        "observation_id": "x", "signal_date": "2026-08-17", "symbol": "ABC",
        "captured_at_utc": "2026-08-18T03:01:00Z", "captured_before_entry": False,
        "event_day_halt": False, "reverse_split_within_30d": False,
        "reverse_split_within_90d": False, "reverse_split_within_180d": False,
        "reverse_split_within_365d": False,
    }])
    result = normalize_capture_flags(frame)
    assert result.loc[0, "captured_at_utc"] == "2026-08-18T03:01:00Z"
    assert result.loc[0, "captured_before_entry"]
