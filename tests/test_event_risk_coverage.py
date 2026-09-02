import pandas as pd

from src.event_risk_coverage import annotate_candidates, normalize_actions, parse_nasdaq_halt_rss


def test_normalize_actions_uses_old_symbol_for_name_change() -> None:
    source = pd.DataFrame(
        [{"action_type": "name_changes", "old_symbol": "OLD", "process_date": "2024-01-02", "id": "1"}]
    )
    result = normalize_actions(source)
    assert result.loc[0, "symbol"] == "OLD"
    assert result.loc[0, "effective_date"] == pd.Timestamp("2024-01-02")


def test_parse_nasdaq_halt_rss_tolerates_malformed_surrounding_xml() -> None:
    payload = b"""not-valid<item><ndaq:IssueSymbol>SMX</ndaq:IssueSymbol>
    <ndaq:ReasonCode>LUDP</ndaq:ReasonCode><ndaq:HaltDate>09/13/2024</ndaq:HaltDate>
    <ndaq:HaltTime>10:01:00</ndaq:HaltTime></item><broken>"""
    result = parse_nasdaq_halt_rss(payload)
    assert result.loc[0, "symbol"] == "SMX"
    assert result.loc[0, "reason_code"] == "LUDP"
    assert result.loc[0, "halt_date"] == pd.Timestamp("2024-09-13")


def test_annotate_candidates_is_backward_looking() -> None:
    candidates = pd.DataFrame(
        [{"symbol": "SMX", "event_date": "2024-09-13", "period": "test"}]
    )
    actions = pd.DataFrame(
        [
            {"symbol": "SMX", "action_type": "reverse_splits", "effective_date": pd.Timestamp("2024-07-15")},
            {"symbol": "SMX", "action_type": "reverse_splits", "effective_date": pd.Timestamp("2024-10-01")},
        ]
    )
    halts = pd.DataFrame([{"symbol": "SMX", "halt_date": pd.Timestamp("2024-09-13")}])
    result = annotate_candidates(candidates, actions, halts)
    assert result.loc[0, "reverse_split_within_90d"]
    assert result.loc[0, "event_day_halt"]
