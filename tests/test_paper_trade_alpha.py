from __future__ import annotations

import pandas as pd

from src.paper_trade_alpha import build_order_blotter


def sample_signals() -> pd.DataFrame:
    return pd.DataFrame([
        {"signal_date": pd.Timestamp("2026-07-24"), "candidate_rank": 1, "candidate_key": "a", "family": "gap_fade", "direction": "long", "symbol": "AAA", "horizon": 5, "reference_close": 20.0, "avg_dollar_volume_20d": 10_000_000.0, "relative_volume": 2.0},
        {"signal_date": pd.Timestamp("2026-07-24"), "candidate_rank": 2, "candidate_key": "b", "family": "momentum", "direction": "short", "symbol": "BBB", "horizon": 10, "reference_close": 40.0, "avg_dollar_volume_20d": 8_000_000.0, "relative_volume": 3.0},
    ])


def test_blotter_is_review_only_and_never_live() -> None:
    orders, rejected = build_order_blotter(sample_signals(), {"paper_capital": 100000, "position_fraction": 0.02, "maximum_orders_per_day": 3, "stop_loss": 0.25})
    assert rejected.empty
    assert len(orders) == 2
    assert orders["status"].eq("REVIEW_REQUIRED").all()
    assert (~orders["live_submission_enabled"]).all()
    assert set(orders["order_type"]) == {"MOO"}


def test_position_sizing_and_directional_stops() -> None:
    orders, _ = build_order_blotter(sample_signals(), {"paper_capital": 100000, "position_fraction": 0.02, "maximum_orders_per_day": 3, "stop_loss": 0.25})
    long_order = orders.loc[orders["symbol"].eq("AAA")].iloc[0]
    short_order = orders.loc[orders["symbol"].eq("BBB")].iloc[0]
    assert long_order["shares"] == 100
    assert short_order["shares"] == 50
    assert long_order["reference_stop_price"] == 15.0
    assert short_order["reference_stop_price"] == 50.0


def test_daily_order_cap_rejects_excess_orders() -> None:
    signals = pd.concat([sample_signals(), sample_signals().assign(symbol=["CCC", "DDD"])], ignore_index=True)
    orders, rejected = build_order_blotter(signals, {"paper_capital": 100000, "position_fraction": 0.02, "maximum_orders_per_day": 2, "stop_loss": 0.25})
    assert len(orders) == 2
    assert len(rejected) == 2
    assert rejected["rejection_reason"].eq("daily_order_cap").all()
