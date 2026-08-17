from __future__ import annotations

import pandas as pd

from src.daily_market_data_updater import merge_prices, normalize_download


def test_normalize_download_creates_canonical_schema() -> None:
    raw = pd.DataFrame({
        "Open": [100.0],
        "High": [105.0],
        "Low": [99.0],
        "Close": [103.0],
        "Adj Close": [103.0],
        "Volume": [1000],
    }, index=pd.DatetimeIndex(["2026-07-27"], name="Date"))

    normalized = normalize_download(raw, "saft")

    assert normalized.columns.tolist() == ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
    assert normalized.iloc[0]["symbol"] == "SAFT"
    assert normalized.iloc[0]["date"] == pd.Timestamp("2026-07-27")
    assert normalized.iloc[0]["close"] == 103.0


def test_merge_prices_replaces_duplicate_symbol_date_with_update() -> None:
    existing = pd.DataFrame([{
        "symbol": "SAFT", "date": pd.Timestamp("2026-07-27"), "open": 100.0,
        "high": 105.0, "low": 99.0, "close": 101.0, "adj_close": 101.0, "volume": 1000,
    }])
    update = pd.DataFrame([{
        "symbol": "SAFT", "date": pd.Timestamp("2026-07-27"), "open": 100.0,
        "high": 106.0, "low": 99.0, "close": 104.0, "adj_close": 104.0, "volume": 1200,
    }])

    merged = merge_prices(existing, update)

    assert len(merged) == 1
    assert merged.iloc[0]["close"] == 104.0
    assert merged.iloc[0]["volume"] == 1200
