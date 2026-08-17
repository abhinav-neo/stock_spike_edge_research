import pandas as pd

from src.alpaca_locate_evidence import broker_etb_evidence, unrecorded_evidence
from src.forward_locate_evidence import append_locate_evidence


def test_only_broker_eligible_etb_shorts_receive_established_locate_evidence() -> None:
    eligibility = pd.DataFrame([
        {"observation_id": "a", "signal_date": "2026-08-17", "symbol": "AAA", "direction": "short", "broker_eligible": True},
        {"observation_id": "b", "signal_date": "2026-08-17", "symbol": "BBB", "direction": "short", "broker_eligible": False},
    ])
    snapshots = pd.DataFrame([
        {"snapshot_date": "2026-08-17", "signal_date": "2026-08-17", "symbol": "AAA", "direction": "short", "borrow_status": "easy_to_borrow", "easy_to_borrow": True},
        {"snapshot_date": "2026-08-17", "signal_date": "2026-08-17", "symbol": "BBB", "direction": "short", "borrow_status": "hard_to_borrow", "easy_to_borrow": False},
    ])
    result = broker_etb_evidence(eligibility, snapshots)
    assert result["observation_id"].tolist() == ["a"]
    assert result.loc[result.index[0], "locate_basis"] == "broker_etb"
    assert bool(result.loc[result.index[0], "locate_confirmed"])


def test_provider_rerun_can_filter_already_recorded_observations() -> None:
    existing = pd.DataFrame([{
        "observation_id": "a", "decision_timestamp": "2026-08-17T00:00:00Z",
        "provider": "alpaca", "locate_requested": False, "locate_confirmed": True,
        "quoted_borrow_rate_annual": 0.0, "available_quantity": pd.NA,
        "source_reference": "alpaca_asset_borrow_status:easy_to_borrow", "locate_basis": "broker_etb",
    }])
    additions = unrecorded_evidence(existing, existing)
    result = append_locate_evidence(existing, additions)
    assert result["observation_id"].tolist() == ["a"]
