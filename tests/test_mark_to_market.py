import pandas as pd
import pytest

from src.mark_to_market import mark_to_market_portfolio, trade_paths, validate_prices


def sample_prices():
    return pd.DataFrame({
        "symbol": ["ABC"] * 4,
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]),
        "open": [10.0, 9.0, 8.0, 7.0],
        "high": [10.5, 9.5, 8.5, 7.5],
        "low": [9.0, 8.0, 7.0, 6.0],
        "close": [9.5, 8.5, 7.5, 6.5],
    })


def test_validate_prices_rejects_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        validate_prices(pd.DataFrame({"symbol": ["ABC"]}))


def test_trade_paths_uses_next_open_and_daily_extremes():
    trades = pd.DataFrame({"symbol": ["ABC"], "event_date": pd.to_datetime(["2024-01-01"]), "horizon": [3]})
    paths, completed = trade_paths(trades, sample_prices())
    assert len(paths) == 3
    assert completed.iloc[0]["entry_price"] == pytest.approx(10.0)
    assert completed.iloc[0]["net_return"] == pytest.approx(0.25)
    assert completed.iloc[0]["mfe"] == pytest.approx(0.30)
    assert completed.iloc[0]["mae"] == pytest.approx(-0.05)


def test_trade_paths_models_gap_through_stop_at_stop_price():
    prices = sample_prices()
    prices.loc[1, "high"] = 15.0
    trades = pd.DataFrame({"symbol": ["ABC"], "event_date": pd.to_datetime(["2024-01-01"]), "horizon": [3]})
    _, completed = trade_paths(trades, prices, stop_loss=0.40)
    assert completed.iloc[0]["exit_reason"] == "stop"
    assert completed.iloc[0]["net_return"] == pytest.approx(-0.40)


def test_mark_to_market_reports_drawdown_and_never_approves_production():
    trades = pd.DataFrame({"trade_id": [0], "entry_date": pd.to_datetime(["2024-01-02"]), "exit_date": pd.to_datetime(["2024-01-04"]), "net_return": [0.25], "mae": [-0.05], "mfe": [0.30]})
    paths = pd.DataFrame({"trade_id": [0, 0, 0], "symbol": ["ABC"] * 3, "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]), "day": [1, 2, 3], "mark_return": [0.05, 0.15, 0.25]})
    equity, summary = mark_to_market_portfolio(paths, trades, initial_capital=100000, position_fraction=0.10)
    assert not equity.empty
    assert summary["ending_equity"] == pytest.approx(102500)
    assert summary["total_return"] == pytest.approx(0.025)
