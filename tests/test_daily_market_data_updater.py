from __future__ import annotations

import pandas as pd
import pytest

from datetime import date, datetime
from zoneinfo import ZoneInfo

from src.daily_market_data_updater import (
    batches,
    discover_symbols,
    latest_weekday,
    latest_completed_session,
    merge_prices,
    normalize_download,
    update_market_data,
)


def test_latest_weekday_leaves_weekdays_and_rolls_back_weekends() -> None:
    assert latest_weekday(date(2026, 8, 14)) == date(2026, 8, 14)
    assert latest_weekday(date(2026, 8, 15)) == date(2026, 8, 14)
    assert latest_weekday(date(2026, 8, 16)) == date(2026, 8, 14)


def test_latest_completed_session_does_not_use_an_open_session() -> None:
    eastern = ZoneInfo("America/New_York")
    before_ready = datetime(2026, 8, 17, 9, 0, tzinfo=eastern)
    after_ready = datetime(2026, 8, 17, 18, 1, tzinfo=eastern)
    assert latest_completed_session(date(2026, 8, 17), before_ready) == date(2026, 8, 14)
    assert latest_completed_session(date(2026, 8, 17), after_ready) == date(2026, 8, 17)


def test_batches_cover_symbols_once_and_validate_size() -> None:
    assert batches(["A", "B", "C", "D", "E"], 2) == [["A", "B"], ["C", "D"], ["E"]]
    with pytest.raises(ValueError, match="positive"):
        batches(["A"], 0)


def test_full_existing_universe_is_not_limited_to_order_symbols(tmp_path) -> None:
    existing = pd.DataFrame({"symbol": ["AAA", "BBB"]})
    orders = tmp_path / "orders.csv"
    orders.write_text("symbol\nAAA\n", encoding="utf-8")
    assert discover_symbols(existing, orders, [], full_existing_universe=True) == ["AAA", "BBB"]


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


def test_failed_batch_retries_symbols_individually(monkeypatch, tmp_path) -> None:
    existing = pd.DataFrame(
        [
            {"symbol": symbol, "date": pd.Timestamp("2026-08-10"), "open": 10.0, "high": 11.0,
             "low": 9.0, "close": 10.0, "adj_close": 10.0, "volume": 1000}
            for symbol in ("AAA", "BBB")
        ]
    )
    output = tmp_path / "prices.parquet"
    existing.to_parquet(output, index=False)
    calls = []

    def fake_download(tickers, **kwargs):
        calls.append(tickers)
        if isinstance(tickers, list):
            raise RuntimeError("transient batch failure")
        return pd.DataFrame(
            {"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.5],
             "Adj Close": [10.5], "Volume": [1200]},
            index=pd.DatetimeIndex(["2026-08-11"], name="Date"),
        )

    monkeypatch.setattr("src.daily_market_data_updater.yf.download", fake_download)
    summary = update_market_data(
        output, tmp_path / "missing-orders.csv", [], date(2026, 8, 11), batch_size=100,
        full_existing_universe=True,
    )

    assert calls == [["AAA", "BBB"], "AAA", "BBB"]
    assert summary["new_rows"] == 2
    assert summary["failed_symbols"] == []


def test_weekend_run_skips_symbols_current_through_friday(monkeypatch, tmp_path) -> None:
    existing = pd.DataFrame(
        [{"symbol": "AAA", "date": pd.Timestamp("2026-08-14"), "open": 10.0, "high": 11.0,
          "low": 9.0, "close": 10.0, "adj_close": 10.0, "volume": 1000}]
    )
    output = tmp_path / "prices.parquet"
    existing.to_parquet(output, index=False)

    def unexpected_download(*args, **kwargs):
        raise AssertionError("a current symbol must not be downloaded on a weekend")

    monkeypatch.setattr("src.daily_market_data_updater.yf.download", unexpected_download)
    summary = update_market_data(
        output, tmp_path / "missing-orders.csv", [], date(2026, 8, 16), full_existing_universe=True
    )
    assert summary["new_rows"] == 0
    assert summary["effective_end_date"] == "2026-08-14"
