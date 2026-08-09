import os

import pandas as pd
import pytest

from src.alpaca_historical_data import (
    AlpacaHistoricalClient,
    coverage_metrics,
    credentials_from_environment,
    regular_session_quotes,
)


class Response:
    status_code = 200
    headers = {}

    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class Session:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.headers = {}
        self.calls = []

    def get(self, url, params, timeout):
        self.calls.append((url, params.copy(), timeout))
        return Response(next(self.payloads))


def test_paginated_quotes_are_normalized_and_authenticated():
    pages = [
        {"quotes": [{"t": "2024-01-02T14:30:00Z", "bp": 10, "ap": 10.01, "bs": 100, "as": 200}], "next_page_token": "next"},
        {"quotes": [{"t": "2024-01-02T14:30:01Z", "bp": 10.01, "ap": 10.02, "bs": 150, "as": 250}], "next_page_token": None},
    ]
    session = Session(pages)
    client = AlpacaHistoricalClient("key", "secret", session=session)
    result = client.quotes("AAPL", pd.Timestamp("2024-01-02", tz="UTC"), pd.Timestamp("2024-01-03", tz="UTC"))
    assert len(result) == 2
    assert session.calls[1][1]["page_token"] == "next"
    assert session.headers["APCA-API-KEY-ID"] == "key"
    assert str(result["timestamp"].dtype) == "datetime64[ns, UTC]"


def test_regular_session_filter_uses_new_york_time():
    quotes = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02 14:29Z", "2024-01-02 14:30Z", "2024-01-02 21:00Z"]),
            "symbol": ["A"] * 3,
        }
    )
    result = regular_session_quotes(quotes)
    assert result["timestamp"].tolist() == [pd.Timestamp("2024-01-02 14:30Z")]


def test_coverage_reports_spread_and_days():
    quotes = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-02 14:30Z", "2024-01-03 14:30Z"]),
            "symbol": ["A", "A"],
            "bid": [9.99, 9.99],
            "ask": [10.01, 10.01],
            "bid_size": [100, 100],
            "ask_size": [100, 100],
        }
    )
    result = coverage_metrics(quotes)
    assert result["trading_days"] == 2
    assert result["median_spread_bps"] == pytest.approx(20)


def test_missing_credentials_fail_without_exposing_values(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(RuntimeError, match="APCA_API_KEY_ID"):
        credentials_from_environment()
    assert "APCA_API_SECRET_KEY" not in os.environ
