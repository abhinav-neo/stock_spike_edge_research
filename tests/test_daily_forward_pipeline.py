from __future__ import annotations

import sys

from src.daily_forward_pipeline import commands


def test_forward_pipeline_never_invokes_order_modules() -> None:
    result = commands("2026-08-10", False, ["AAA"])
    flattened = " ".join(part for command in result for part in command)
    assert result[0] == [
        sys.executable, "-B", "-m", "src.daily_market_data_updater", "--end-date", "2026-08-10",
        "--full-existing-universe", "--symbols", "AAA"
    ]
    assert result[-9] == [sys.executable, "-B", "-m", "src.forward_observation"]
    assert result[-8] == [sys.executable, "-B", "-m", "src.forward_event_risk"]
    assert result[-1][0:4] == [
        sys.executable, "-B", "-m", "src.forward_breakthrough_assessment"
    ]
    assert "reports/forward_observation/execution_evaluation_sip_v1.csv" in result[-1]
    assert result[-7] == [
        sys.executable, "-B", "-m", "src.alpaca_operational_snapshot", "--date", "2026-08-10"
    ]
    assert result[-6] == [sys.executable, "-B", "-m", "src.forward_eligibility"]
    assert result[-5] == [sys.executable, "-B", "-m", "src.alpaca_locate_evidence"]
    assert result[-4] == [
        sys.executable, "-B", "-m", "src.alpaca_account_snapshot", "--date", "2026-08-10"
    ]
    assert result[-3][0:4] == [sys.executable, "-B", "-m", "src.forward_quote_capture"]
    assert result[-3][4:8] == ["--feed", "sip", "--minimum-entry-date", "2026-09-02"]
    assert result[-2][0:4] == [sys.executable, "-B", "-m", "src.forward_execution_evaluation"]
    assert "data/raw/forward_quotes_sip_v1" in result[-2]
    assert "paper_trade_alpha" not in flattened
    assert "paper_fill_tracker" not in flattened


def test_skip_update_runs_observer_only() -> None:
    assert commands("2026-08-10", True, []) == [
        [sys.executable, "-B", "-m", "src.forward_observation"],
        [sys.executable, "-B", "-m", "src.forward_event_risk"],
        [sys.executable, "-B", "-m", "src.alpaca_operational_snapshot", "--date", "2026-08-10"],
        [sys.executable, "-B", "-m", "src.forward_eligibility"],
        [sys.executable, "-B", "-m", "src.alpaca_locate_evidence"],
        [sys.executable, "-B", "-m", "src.alpaca_account_snapshot", "--date", "2026-08-10"],
        [sys.executable, "-B", "-m", "src.forward_quote_capture", "--feed", "sip",
         "--minimum-entry-date", "2026-09-02", "--output", "data/raw/forward_quotes_sip_v1",
         "--summary", "reports/forward_observation/quote_coverage_sip_v1.csv"],
        [sys.executable, "-B", "-m", "src.forward_execution_evaluation",
         "--minimum-entry-date", "2026-09-02", "--quotes", "data/raw/forward_quotes_sip_v1",
         "--output", "reports/forward_observation/execution_evaluation_sip_v1.csv"],
        [sys.executable, "-B", "-m", "src.forward_breakthrough_assessment",
         "--executions", "reports/forward_observation/execution_evaluation_sip_v1.csv"],
    ]
