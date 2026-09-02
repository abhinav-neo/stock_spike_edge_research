from __future__ import annotations

import pandas as pd

from src.forward_breakthrough_assessment import combined_verdict


def evidence_frames():
    ledger = pd.DataFrame([{
        "signal_date": "2026-08-10", "candidate_key": "locked", "symbol": "AAA", "direction": "short",
        "observation_status": "SETTLED"
    }])
    snapshots = pd.DataFrame([{
        "signal_date": "2026-08-10", "symbol": "AAA", "direction": "short", "shortable": True,
        "easy_to_borrow": True,
    }])
    executions = pd.DataFrame([{"entry_spread_bps": 10.0, "exit_spread_bps": 12.0}])
    accounts = pd.DataFrame([{
        "snapshot_date": "2026-08-10", "status": "ACTIVE", "shorting_enabled": True,
        "trading_blocked": False, "account_blocked": False, "trade_suspended_by_user": False,
    }])
    eligibility = pd.DataFrame([{
        "observation_id": "placeholder", "broker_eligible": True, "shortable": True, "easy_to_borrow": True,
    }])
    from src.forward_quote_capture import observation_id
    eligibility.loc[0, "observation_id"] = observation_id(ledger.iloc[0])
    executions["observation_id"] = eligibility.loc[0, "observation_id"]
    locates = pd.DataFrame([{
        "observation_id": eligibility.loc[0, "observation_id"], "locate_confirmed": True,
    }])
    return ledger, snapshots, executions, accounts, eligibility, locates


def test_breakthrough_requires_both_statistical_and_operational_gates() -> None:
    ledger, snapshots, executions, accounts, eligibility, locates = evidence_frames()
    result = combined_verdict(
        {"statistical_gate_passed": True}, ledger, snapshots, executions, {}, accounts, eligibility, locates,
        economics={"economic_gate_passed": True},
    )
    assert result["operational_gate_passed"] is True
    assert result["breakthrough"] is True
    assert result["verdict"] == "BREAKTHROUGH"


def test_missing_economic_gate_blocks_breakthrough() -> None:
    ledger, snapshots, executions, accounts, eligibility, locates = evidence_frames()
    result = combined_verdict(
        {"statistical_gate_passed": True}, ledger, snapshots, executions, {}, accounts, eligibility, locates
    )
    assert result["operational_gate_passed"] is True
    assert result["economic_gate_passed"] is False
    assert result["breakthrough"] is False


def test_missing_actual_locates_blocks_operational_gate() -> None:
    ledger, snapshots, executions, accounts, eligibility, _ = evidence_frames()
    result = combined_verdict({"statistical_gate_passed": True}, ledger, snapshots, executions, {}, accounts, eligibility)
    assert result["operational_gate_passed"] is False
    assert result["breakthrough"] is False


def test_blocked_account_prevents_operational_promotion() -> None:
    ledger, snapshots, executions, accounts, eligibility, locates = evidence_frames()
    accounts.loc[0, "trading_blocked"] = True
    result = combined_verdict(
        {"statistical_gate_passed": True}, ledger, snapshots, executions, {}, accounts, eligibility, locates
    )
    assert result["account_controls_ready"] is False
    assert result["operational_gate_passed"] is False


def test_execution_for_rejected_signal_fails_integrity_gate() -> None:
    ledger, snapshots, executions, accounts, eligibility, locates = evidence_frames()
    eligibility.loc[0, "broker_eligible"] = False
    result = combined_verdict(
        {"statistical_gate_passed": True}, ledger, snapshots, executions, {}, accounts, eligibility, locates
    )
    assert result["integrity_gate_passed"] is False
    assert result["rejected_signal_executions"] == 1
    assert result["operational_gate_passed"] is False


def test_unknown_locate_decision_fails_integrity_gate() -> None:
    ledger, snapshots, executions, accounts, eligibility, locates = evidence_frames()
    locates.loc[0, "observation_id"] = "not-in-ledger"
    result = combined_verdict(
        {"statistical_gate_passed": True}, ledger, snapshots, executions, {}, accounts, eligibility, locates
    )
    assert result["integrity_gate_passed"] is False
    assert result["unknown_locate_decisions"] == 1


def test_pre_protocol_settlement_is_not_required_execution_coverage() -> None:
    ledger, snapshots, executions, accounts, eligibility, locates = evidence_frames()
    ledger["entry_date"] = "2026-09-01"
    executions = executions.iloc[0:0]
    result = combined_verdict(
        {"statistical_gate_passed": False}, ledger, snapshots, executions, {}, accounts,
        eligibility, locates, pd.Timestamp("2026-09-02"),
    )
    assert result["execution_coverage"] == 0.0
    assert result["integrity_gate_passed"] is True
    assert result["operational_gate_passed"] is False
