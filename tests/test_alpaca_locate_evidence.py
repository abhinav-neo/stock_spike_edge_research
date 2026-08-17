import pandas as pd

from src.alpaca_locate_evidence import broker_etb_evidence


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
