import pandas as pd
import pytest

from src.stress_test_mtm import scenario_grid, temporal_stability, yearly_trade_summary


def test_scenario_grid_builds_cartesian_product():
    cfg = {"borrow_bps_grid": [0, 1000], "locate_probability_grid": [1.0, 0.5]}
    assert scenario_grid(cfg) == [(0.0, 1.0), (0.0, 0.5), (1000.0, 1.0), (1000.0, 0.5)]


def test_yearly_trade_summary_and_stability():
    trades = pd.DataFrame(
        {
            "entry_date": pd.to_datetime(["2023-01-03", "2023-05-01", "2024-02-01"]),
            "net_return": [0.20, -0.10, 0.30],
            "exit_reason": ["time", "stop", "time"],
            "stop_fill_type": ["none", "intraday_stop", "none"],
        }
    )
    yearly = yearly_trade_summary(trades, stake=5000)
    assert list(yearly["entry_year"]) == [2023, 2024]
    assert yearly.loc[yearly["entry_year"] == 2023, "total_pnl"].iloc[0] == pytest.approx(500)
    stability = temporal_stability(yearly)
    assert stability["years"] == 2
    assert stability["positive_year_fraction"] == pytest.approx(1.0)
    assert stability["profitable_year_fraction"] == pytest.approx(1.0)
