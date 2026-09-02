from __future__ import annotations

from datetime import date

import pandas as pd

from src.alpaca_operational_snapshot import AlpacaAssetClient, append_snapshots, collect_snapshots


class Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {
            "id": "asset-1", "status": "active", "tradable": True, "shortable": True,
            "easy_to_borrow": False, "fractionable": True, "maintenance_margin_requirement": 30,
        }


class Session:
    def __init__(self):
        self.headers = {}
        self.urls = []

    def get(self, url, timeout):
        self.urls.append((url, timeout))
        return Response()


def test_snapshot_is_read_only_and_records_shortability() -> None:
    ledger = pd.DataFrame([{
        "signal_date": "2026-08-10", "symbol": "AAA", "direction": "short", "observation_status": "OPEN"
    }])
    session = Session()
    result = collect_snapshots(ledger, AlpacaAssetClient("key", "secret", session), date(2026, 8, 10))
    assert result.iloc[0]["shortable"]
    assert not result.iloc[0]["easy_to_borrow"]
    assert session.urls == [("https://paper-api.alpaca.markets/v2/assets/AAA", 30)]
    assert set(session.headers) == {"APCA-API-KEY-ID", "APCA-API-SECRET-KEY"}


def test_snapshot_append_is_idempotent() -> None:
    row = pd.DataFrame([{
        "snapshot_date": "2026-08-10", "signal_date": "2026-08-10", "symbol": "AAA", "direction": "short"
    }])
    assert len(append_snapshots(row, row)) == 1


def test_snapshot_append_retains_daily_borrowability_changes() -> None:
    first = pd.DataFrame([{
        "snapshot_date": "2026-08-10", "signal_date": "2026-08-10", "symbol": "AAA", "direction": "short",
        "easy_to_borrow": True,
    }])
    second = first.assign(snapshot_date="2026-08-11", easy_to_borrow=False)
    result = append_snapshots(first, second)
    assert len(result) == 2
    assert result["easy_to_borrow"].tolist() == [True, False]
