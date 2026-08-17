import pandas as pd
import pytest

from src.forward_locate_evidence import append_locate_evidence, validate_locate_evidence


def evidence(observation_id: str = "obs-1") -> pd.DataFrame:
    return pd.DataFrame([{
        "observation_id": observation_id,
        "decision_timestamp": "2026-08-17T14:00:00Z",
        "provider": "broker-export",
        "locate_requested": True,
        "locate_confirmed": True,
        "quoted_borrow_rate_annual": 0.25,
        "available_quantity": 100,
        "source_reference": "redacted-reference",
    }])


def test_valid_actual_locate_is_normalized() -> None:
    result = validate_locate_evidence(evidence())
    assert str(result.loc[0, "decision_timestamp"].tz) == "UTC"
    assert bool(result.loc[0, "locate_confirmed"])


def test_confirmation_requires_request_and_quantity() -> None:
    frame = evidence()
    frame.loc[0, "locate_requested"] = False
    with pytest.raises(ValueError, match="not requested"):
        validate_locate_evidence(frame)
    frame = evidence()
    frame.loc[0, "available_quantity"] = 0
    with pytest.raises(ValueError, match="positive"):
        validate_locate_evidence(frame)


def test_append_rejects_duplicate_decisions() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        append_locate_evidence(evidence(), evidence())
