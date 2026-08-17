import pandas as pd
import pytest

from src.mark_to_market import (
    apply_locate_model,
    margin_liquidation_price_multiple,
    mark_to_market_portfolio,
    trade_paths,
    validate_prices,
)


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


def test_trade_paths_charges_annual_borrow_cost():
    trades = pd.DataFrame({"symbol": ["ABC"], "event_date": pd.to_datetime(["2024-01-01"]), "horizon": [3]})
    paths, completed = trade_paths(trades, sample_prices(), short_borrow_bps_annual=2520)
    assert completed.iloc[0]["gross_return"] == pytest.approx(0.25)
    assert completed.iloc[0]["borrow_cost_return"] == pytest.approx(0.003)
    assert completed.iloc[0]["net_return"] == pytest.approx(0.247)
    assert paths.iloc[-1]["accrued_borrow_return"] == pytest.approx(0.003)


def test_trade_paths_intraday_stop_fills_at_stop_price():
    prices = sample_prices()
    prices.loc[1, "high"] = 15.0
    trades = pd.DataFrame({"symbol": ["ABC"], "event_date": pd.to_datetime(["2024-01-01"]), "horizon": [3]})
    _, completed = trade_paths(trades, prices, stop_loss=0.40)
    assert completed.iloc[0]["exit_reason"] == "stop"
    assert completed.iloc[0]["stop_fill_type"] == "intraday_stop"
    assert completed.iloc[0]["net_return"] == pytest.approx(-0.40)


def test_trade_paths_gap_through_stop_fills_at_open():
    prices = sample_prices()
    prices.loc[1, "open"] = 16.0
    prices.loc[1, "high"] = 17.0
    prices.loc[1, "low"] = 15.0
    prices.loc[1, "close"] = 16.5
    trades = pd.DataFrame({"symbol": ["ABC"], "event_date": pd.to_datetime(["2024-01-01"]), "horizon": [3]})
    _, completed = trade_paths(trades, prices, stop_loss=0.40)
    assert completed.iloc[0]["exit_reason"] == "stop"
    assert completed.iloc[0]["stop_fill_type"] == "gap_open"
    assert completed.iloc[0]["exit_price"] == pytest.approx(16.0)
    assert completed.iloc[0]["net_return"] == pytest.approx(-0.60)


def test_margin_liquidation_threshold_and_gap_aware_fill():
    assert margin_liquidation_price_multiple(0.50, 0.30) == pytest.approx(1.5 / 1.3)
    prices = sample_prices()
    prices.loc[1, ["open", "high", "low", "close"]] = [12.0, 12.5, 11.5, 12.0]
    trades = pd.DataFrame({"symbol": ["ABC"], "event_date": pd.to_datetime(["2024-01-01"]), "horizon": [3]})
    _, completed = trade_paths(
        trades,
        prices,
        initial_margin_requirement=0.50,
        maintenance_margin_requirement=0.30,
    )
    assert completed.iloc[0]["exit_reason"] == "margin_liquidation"
    assert completed.iloc[0]["stop_fill_type"] == "gap_open"
    assert completed.iloc[0]["exit_price"] == pytest.approx(12.0)
    assert completed.iloc[0]["net_return"] == pytest.approx(-0.20)


def test_margin_parameters_must_be_complete_and_non_negative():
    trades = pd.DataFrame({"symbol": ["ABC"], "event_date": pd.to_datetime(["2024-01-01"]), "horizon": [3]})
    with pytest.raises(ValueError, match="supplied together"):
        trade_paths(trades, sample_prices(), initial_margin_requirement=0.50)
    with pytest.raises(ValueError, match="non-negative"):
        margin_liquidation_price_multiple(-0.1, 0.30)


def test_locate_model_is_reproducible_and_can_reject_trades():
    trades = pd.DataFrame({
        "symbol": ["ABC", "DEF", "GHI", "JKL"],
        "event_date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        "horizon": [20, 20, 20, 20],
    })
    accepted1, rejected1 = apply_locate_model(trades, 0.5, 42)
    accepted2, rejected2 = apply_locate_model(trades, 0.5, 42)
    pd.testing.assert_frame_equal(accepted1, accepted2)
    pd.testing.assert_frame_equal(rejected1, rejected2)
    assert len(accepted1) + len(rejected1) == len(trades)


def test_mark_to_market_avoids_exit_date_double_counting_and_reports_utilization():
    trades = pd.DataFrame({
        "trade_id": [0],
        "entry_date": pd.to_datetime(["2024-01-02"]),
        "exit_date": pd.to_datetime(["2024-01-04"]),
        "net_return": [0.25],
        "borrow_cost_return": [0.0],
        "mae": [-0.05],
        "mfe": [0.30],
        "exit_reason": ["time"],
        "stop_fill_type": ["none"],
    })
    paths = pd.DataFrame({
        "trade_id": [0, 0, 0],
        "symbol": ["ABC"] * 3,
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "day": [1, 2, 3],
        "mark_return": [0.05, 0.15, 0.25],
    })
    equity, summary = mark_to_market_portfolio(paths, trades, initial_capital=100000, position_fraction=0.10)
    assert not equity.empty
    assert equity.iloc[-1]["equity"] == pytest.approx(102500)
    assert summary["ending_equity"] == pytest.approx(102500)
    assert summary["total_return"] == pytest.approx(0.025)
    assert summary["active_days_pct"] == pytest.approx(1.0)
    assert summary["average_gross_exposure"] == pytest.approx(0.10)
    assert summary["maximum_concurrent_positions"] == 1
    assert "active_day_sharpe" in summary
    assert "annualized_return_on_deployed_capital" in summary
