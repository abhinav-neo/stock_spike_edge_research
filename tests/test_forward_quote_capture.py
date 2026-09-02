from __future__ import annotations

import pandas as pd

from src.forward_quote_capture import capture_available_windows, observation_id, quote_window


class Client:
    def __init__(self):
        self.calls = []

    def quotes(self, symbol, start, end, feed):
        self.calls.append((symbol, start, end, feed))
        return pd.DataFrame({
            "symbol": [symbol], "timestamp": [start], "bid": [10.0], "ask": [10.01],
            "bid_size": [100], "ask_size": [200],
        })


def row() -> pd.DataFrame:
    return pd.DataFrame([{
        "signal_date": "2026-08-10", "candidate_key": "locked", "symbol": "AAA", "direction": "short",
        "entry_date": "2026-08-11", "exit_date": "2026-08-17",
    }])


def test_quote_windows_use_new_york_session_and_dst() -> None:
    entry_start, entry_end = quote_window(pd.Timestamp("2026-08-11"), "entry")
    exit_start, exit_end = quote_window(pd.Timestamp("2026-08-17"), "exit")
    assert entry_start == pd.Timestamp("2026-08-11T13:30:00Z")
    assert entry_end == pd.Timestamp("2026-08-11T13:35:00Z")
    assert exit_start == pd.Timestamp("2026-08-17T19:55:00Z")
    assert exit_end == pd.Timestamp("2026-08-17T20:00:00Z")


def test_capture_is_partitioned_and_resumable(tmp_path) -> None:
    client = Client()
    first = capture_available_windows(row(), client, tmp_path)
    assert len(first) == 2
    assert first["status"].eq("downloaded").all()
    assert len(client.calls) == 2
    second = capture_available_windows(row(), client, tmp_path)
    assert second["status"].eq("reused").all()
    assert len(client.calls) == 2
    assert (tmp_path / f"observation={observation_id(row().iloc[0])}" / "phase=entry" / "quotes.parquet").exists()


def test_capture_skips_broker_ineligible_observations(tmp_path) -> None:
    client = Client()
    assert capture_available_windows(row(), client, tmp_path, eligible_ids=set()).empty
    assert client.calls == []


def test_capture_enforces_prospective_protocol_start_and_records_feed(tmp_path) -> None:
    client = Client()
    ledger = pd.concat([
        row().assign(entry_date="2026-09-01"),
        row().assign(signal_date="2026-09-01", entry_date="2026-09-02", exit_date="2026-09-08"),
    ], ignore_index=True)
    result = capture_available_windows(
        ledger, client, tmp_path, feed="sip", minimum_entry_date=pd.Timestamp("2026-09-02")
    )
    assert len(result) == 2
    assert result["feed"].eq("sip").all()
    assert {call[3] for call in client.calls} == {"sip"}
