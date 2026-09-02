import pandas as pd

from src.halt_exclusion_research import validation_gate


def test_validation_gate_does_not_consume_test_outcomes() -> None:
    candidates = pd.DataFrame(
        [
            {"period": "validation", "event_day_halt": False},
            {"period": "validation", "event_day_halt": True},
            {"period": "test", "event_day_halt": False},
        ]
    )
    result = validation_gate(candidates, minimum_events=2)
    assert result["validation_candidates"] == 2
    assert result["validation_candidates_retained"] == 1
    assert not result["validation_sample_gate_passed"]
    assert not result["test_evaluation_authorized"]
