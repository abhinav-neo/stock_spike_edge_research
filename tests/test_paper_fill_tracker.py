from __future__ import annotations

import pandas as pd

from src.paper_fill_tracker import process_orders, mark_positions


def _orders() -> pd.DataFrame:
    return pd.DataFrame([{
        "order_id": "PAPER-20260724-001",
        "signal_date": pd.Timestamp("2026-07-24"),
        "symbol": "SAFT",
        "side": "SELL_SHORT",
        "direction": "short",
        "shares": 19,
        "reference_close": 100.0,
        "reference_stop_price": 125.0,
        "live_submission_enabled": False,
    }])


def test_order_remains_pending_without_next_session() -> None:
    prices = pd.DataFrame([{"symbol": "SAFT", "date": pd.Timestamp("2026-07-24"), "open": 99.0, "close": 100.0}])
    fills, pending = process_orders(_orders(), prices, {"stop_loss": 0.25, "paper_fill_slippage_bps": 5})
    assert fills.empty
    assert len(pending) == 1
    assert pending.iloc[0]["fill_status"] == "PENDING_NEXT_SESSION"
    assert not bool(pending.iloc[0]["live_submission_enabled"])


def test_short_fill_uses_adverse_slippage_and_rebased_stop() -> None:
    prices = pd.DataFrame([
        {"symbol": "SAFT", "date": pd.Timestamp("2026-07-24"), "open": 99.0, "close": 100.0},
        {"symbol": "SAFT", "date": pd.Timestamp("2026-07-27"), "open": 102.0, "close": 101.0},
    ])
    fills, pending = process_orders(_orders(), prices, {"stop_loss": 0.25, "paper_fill_slippage_bps": 5})
    assert pending.empty
    assert len(fills) == 1
    fill = fills.iloc[0]
    assert fill["fill_price"] == 102.0 * (1.0 - 0.0005)
    assert fill["active_stop_price"] == fill["fill_price"] * 1.25
    assert fill["fill_status"] == "PAPER_FILLED"


def test_short_mark_to_market_profit_when_price_falls() -> None:
    fills = pd.DataFrame([{
        "order_id": "PAPER-20260724-001",
        "symbol": "SAFT",
        "direction": "short",
        "shares": 19,
        "fill_date": pd.Timestamp("2026-07-27"),
        "fill_price": 100.0,
        "fill_notional": 1900.0,
        "active_stop_price": 125.0,
    }])
    prices = pd.DataFrame([
        {"symbol": "SAFT", "date": pd.Timestamp("2026-07-27"), "close": 100.0},
        {"symbol": "SAFT", "date": pd.Timestamp("2026-07-28"), "close": 90.0},
    ])
    positions = mark_positions(fills, prices)
    assert positions.iloc[0]["gross_return"] == 0.10
    assert positions.iloc[0]["unrealized_pnl"] == 190.0
    assert not bool(positions.iloc[0]["live_submission_enabled"])
