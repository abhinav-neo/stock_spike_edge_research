import pandas as pd
import pytest

from src.v5_mtm_research import apply_round_trip_cost, attach_event_dates, daily_equity_curve


def test_attach_event_dates_uses_symbol_and_entry_date():
    trades = pd.DataFrame({"symbol": ["AAA"], "entry_date": ["2024-01-03"]})
    features = pd.DataFrame({"symbol": ["AAA"], "event_date": ["2024-01-02"], "entry_date": ["2024-01-03"]})
    result = attach_event_dates(trades, features, 5)
    assert result.loc[0, "event_date"] == pd.Timestamp("2024-01-02")
    assert result.loc[0, "horizon"] == 5


def test_round_trip_cost_is_applied_once_to_marks_and_completion():
    paths = pd.DataFrame({"mark_return": [0.10, 0.20]})
    trades = pd.DataFrame({"net_return": [0.20]})
    adjusted_paths, adjusted_trades = apply_round_trip_cost(paths, trades, 100)
    assert adjusted_paths["mark_return"].tolist() == pytest.approx([0.09, 0.19])
    assert adjusted_trades["net_return"].tolist() == pytest.approx([0.19])


def test_daily_equity_includes_flat_calendar_days_and_unrealized_loss():
    paths = pd.DataFrame({"trade_id": [0, 0], "date": pd.to_datetime(["2024-01-02", "2024-01-04"]), "mark_return": [-0.10, 0.20]})
    trades = pd.DataFrame({"trade_id": [0], "exit_date": pd.to_datetime(["2024-01-04"]), "net_return": [0.20]})
    calendar = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    equity = daily_equity_curve(paths, trades, calendar, 100_000, 10_000)
    assert equity["equity"].tolist() == pytest.approx([99_000, 100_000, 102_000])
    assert equity.loc[0, "drawdown"] == 0.0
