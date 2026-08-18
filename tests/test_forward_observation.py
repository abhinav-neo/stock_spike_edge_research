from __future__ import annotations

import pandas as pd
import pytest

from src.forward_observation import (
    LEDGER_COLUMNS,
    append_signals,
    enforce_model_lock,
    estimated_open_exit_dates,
    evaluate_evidence,
    model_fingerprint,
    settle_observations,
)


def signal() -> pd.DataFrame:
    return pd.DataFrame([{
        "signal_date": "2026-08-03", "candidate_rank": 1, "candidate_key": "locked-a",
        "family": "gap_fade", "direction": "short", "symbol": "AAA", "horizon": 2,
        "reference_close": 10.0, "avg_dollar_volume_20d": 5_000_000.0, "relative_volume": 3.0,
    }])


def test_append_is_idempotent_and_zero_allocation() -> None:
    first = append_signals(pd.DataFrame(), signal())
    second = append_signals(first, signal())
    assert len(second) == 1
    assert second.iloc[0]["allocation_fraction"] == 0.0
    assert not bool(second.iloc[0]["orders_enabled"])
    assert second.iloc[0]["observation_status"] == "OPEN"


def test_empty_ledger_retains_schema() -> None:
    ledger = append_signals(pd.DataFrame(), pd.DataFrame())
    assert ledger.empty
    assert ledger.columns.tolist() == LEDGER_COLUMNS


def test_model_lock_detects_candidate_or_configuration_drift(tmp_path) -> None:
    selected = tmp_path / "selected.csv"
    selected.write_text("candidate\nlocked-a\n", encoding="utf-8")
    lock_path = tmp_path / "lock.json"
    fingerprint = model_fingerprint(selected, {"round_trip_cost_bps": 100})
    created = enforce_model_lock(lock_path, fingerprint, "2026-08-10")
    assert created["model_locked"] is True
    enforce_model_lock(lock_path, fingerprint, "2026-08-10")

    selected.write_text("candidate\nchanged\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Locked forward model changed"):
        enforce_model_lock(lock_path, model_fingerprint(selected, {"round_trip_cost_bps": 100}), "2026-08-10")


def test_observation_settles_only_after_horizon_is_available() -> None:
    ledger = append_signals(pd.DataFrame(), signal())
    incomplete = pd.DataFrame([
        {"symbol": "AAA", "date": "2026-08-04", "open": 11.0, "close": 10.5},
    ])
    assert settle_observations(ledger, incomplete, 100).iloc[0]["observation_status"] == "OPEN"

    complete = pd.concat([incomplete, pd.DataFrame([
        {"symbol": "AAA", "date": "2026-08-05", "open": 10.5, "close": 9.0},
    ])], ignore_index=True)
    settled = settle_observations(ledger, complete, 100)
    row = settled.iloc[0]
    assert row["observation_status"] == "SETTLED"
    assert row["gross_return"] == 1 - 9.0 / 11.0
    assert row["net_return"] == row["gross_return"] - 0.01
    assert pd.Timestamp(row["entry_date"]) == pd.Timestamp("2026-08-04")
    assert pd.Timestamp(row["exit_date"]) == pd.Timestamp("2026-08-05")


def test_observation_settles_after_csv_reload(tmp_path) -> None:
    ledger_path = tmp_path / "ledger.csv"
    append_signals(pd.DataFrame(), signal()).to_csv(ledger_path, index=False)
    reloaded = pd.read_csv(ledger_path)
    prices = pd.DataFrame([
        {"symbol": "AAA", "date": "2026-08-04", "open": 11.0, "close": 10.5},
        {"symbol": "AAA", "date": "2026-08-05", "open": 10.5, "close": 9.0},
    ])

    settled = settle_observations(reloaded, prices, 100)

    assert settled.iloc[0]["observation_status"] == "SETTLED"
    assert settled.iloc[0]["entry_date"] == pd.Timestamp("2026-08-04")


def test_observation_settles_with_arrow_string_date_columns() -> None:
    ledger = append_signals(pd.DataFrame(), signal()).convert_dtypes(dtype_backend="pyarrow")
    prices = pd.DataFrame([
        {"symbol": "AAA", "date": "2026-08-04", "open": 11.0, "close": 10.5},
        {"symbol": "AAA", "date": "2026-08-05", "open": 10.5, "close": 9.0},
    ])

    settled = settle_observations(ledger, prices, 100)

    assert settled.iloc[0]["observation_status"] == "SETTLED"
    assert settled["entry_date"].dtype == "datetime64[ns]"
    assert settled.iloc[0]["entry_date"] == pd.Timestamp("2026-08-04")


def test_evidence_gate_requires_sample_span_confidence_and_candidate_breadth() -> None:
    rows = []
    for index in range(100):
        rows.append({
            "signal_date": pd.Timestamp("2026-08-10") + pd.Timedelta(days=index * 2),
            "candidate_key": f"candidate-{index % 4}",
            "observation_status": "SETTLED",
            "net_return": 0.02 if index % 5 else 0.01,
        })
    result = evaluate_evidence(pd.DataFrame(rows), {
        "minimum_settled_observations": 100,
        "minimum_independent_signal_dates": 60,
        "minimum_calendar_span_days": 180,
        "minimum_positive_candidate_fraction": 0.75,
    })
    assert result["statistical_gate_passed"] is True
    assert result["independent_signal_dates"] == 100
    assert result["operational_gate_passed"] is False
    assert result["breakthrough"] is False


def test_evidence_gate_rejects_short_or_empty_samples() -> None:
    assert evaluate_evidence(pd.DataFrame(), {})["statistical_gate_passed"] is False
    short = pd.DataFrame([{
        "signal_date": "2026-08-10", "candidate_key": "a", "observation_status": "SETTLED", "net_return": 0.5
    }])
    assert evaluate_evidence(short, {})["statistical_gate_passed"] is False


def test_estimated_exit_dates_use_horizon_sessions() -> None:
    ledger = pd.DataFrame([
        {
            "signal_date": "2026-08-10", "entry_date": "2026-08-11", "horizon": 10,
            "observation_status": "OPEN",
        },
        {
            "signal_date": "2026-08-17", "entry_date": pd.NA, "horizon": 10,
            "observation_status": "OPEN",
        },
    ])
    assert estimated_open_exit_dates(ledger) == ("2026-08-24", "2026-08-31")
