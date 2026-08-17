from __future__ import annotations

from src.alpha_research_report import decision_status, fmt_pct


def test_no_locked_survivors_is_rejected() -> None:
    status, action = decision_status(0, 0.50, 0.10, 0.40, 0.25)
    assert status == "REJECTED"
    assert "No hypothesis survived" in action


def test_target_miss_is_not_deployable() -> None:
    status, _ = decision_status(3, 0.25, 0.10, 0.40, 0.25)
    assert status == "RESEARCH EDGE, TARGET MISSED"


def test_excessive_drawdown_fails_even_when_target_met() -> None:
    status, _ = decision_status(3, 0.45, 0.35, 0.40, 0.25)
    assert status == "RETURN TARGET MET, RISK FAILED"


def test_target_and_risk_gate_only_create_paper_candidate() -> None:
    status, action = decision_status(3, 0.45, 0.20, 0.40, 0.25)
    assert status == "PAPER-TRADING CANDIDATE"
    assert "not production approval" in action


def test_percent_format() -> None:
    assert fmt_pct(0.4) == "40.00%"
