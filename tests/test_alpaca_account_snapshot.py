from __future__ import annotations

from datetime import date

import pandas as pd

from src.alpaca_account_snapshot import AlpacaAccountClient, append_account_snapshots, normalize_account


class Response:
    def raise_for_status(self):
        return None

    def json(self):
        return {"id": "must-not-store", "account_number": "must-not-store", "status": "ACTIVE", "cash": "10000",
                "equity": "10000", "buying_power": "20000", "shorting_enabled": True, "pattern_day_trader": False}


class Session:
    def __init__(self):
        self.headers = {}

    def get(self, url, timeout):
        assert url == "https://paper-api.alpaca.markets/v2/account"
        assert timeout == 30
        return Response()


def test_account_snapshot_excludes_identifiers_and_is_read_only() -> None:
    session = Session()
    payload = AlpacaAccountClient("key", "secret", session).account()
    result = normalize_account(payload, date(2026, 8, 10))
    assert "id" not in result
    assert "account_number" not in result
    assert result.iloc[0]["shorting_enabled"]
    assert set(session.headers) == {"APCA-API-KEY-ID", "APCA-API-SECRET-KEY"}


def test_account_snapshot_append_is_one_row_per_day() -> None:
    row = pd.DataFrame([{"snapshot_date": "2026-08-10", "status": "ACTIVE"}])
    assert len(append_account_snapshots(row, row)) == 1
