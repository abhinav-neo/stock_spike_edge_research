import pandas as pd

from src.paper_position_lifecycle import evaluate_position


def fill(direction="long", stop=75.0, horizon=2):
    return pd.Series({"order_id": "P1", "symbol": "AAA", "direction": direction, "shares": 10,
                      "fill_date": pd.Timestamp("2026-01-02"), "fill_price": 100.0,
                      "fill_notional": 1000.0, "active_stop_price": stop, "horizon": horizon})


def bars(rows):
    return pd.DataFrame([{"symbol": "AAA", "date": pd.Timestamp(date), "open": o, "high": h, "low": l, "close": c}
                         for date, o, h, l, c in rows])


def test_long_gap_stop_uses_open() -> None:
    prices = bars([("2026-01-02", 100, 103, 98, 101), ("2026-01-05", 70, 72, 65, 68)])
    result = evaluate_position(fill(stop=75.0, horizon=5), prices, 0.0)
    assert result["position_status"] == "CLOSED"
    assert result["exit_reason"] == "STOP"
    assert result["stop_fill_type"] == "gap_open"
    assert result["exit_price"] == 70.0


def test_short_intraday_stop_uses_stop_price() -> None:
    prices = bars([("2026-01-02", 100, 102, 97, 99), ("2026-01-05", 105, 126, 104, 120)])
    result = evaluate_position(fill(direction="short", stop=125.0, horizon=5), prices, 0.0)
    assert result["position_status"] == "CLOSED"
    assert result["stop_fill_type"] == "intraday"
    assert result["exit_price"] == 125.0
    assert result["realized_return"] == -0.25


def test_time_exit_occurs_at_horizon_close() -> None:
    prices = bars([("2026-01-02", 100, 102, 98, 101), ("2026-01-05", 101, 104, 100, 103),
                   ("2026-01-06", 103, 105, 102, 104)])
    result = evaluate_position(fill(stop=75.0, horizon=2), prices, 0.0)
    assert result["position_status"] == "CLOSED"
    assert result["exit_reason"] == "TIME"
    assert result["exit_price"] == 103.0
