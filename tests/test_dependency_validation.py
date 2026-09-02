import numpy as np
import pandas as pd
import pytest

from src.dependency_validation import (
    cluster_accepted_rules,
    jaccard_similarity,
    monthly_block_bootstrap,
    simulate_portfolio,
)


def test_jaccard_similarity_handles_overlap():
    left = pd.Series([True, True, False, False])
    right = pd.Series([True, False, True, False])
    assert jaccard_similarity(left, right) == pytest.approx(1 / 3)


def test_monthly_block_bootstrap_preserves_positive_edge():
    dates = pd.date_range("2020-01-01", periods=24, freq="MS")
    trades = pd.DataFrame({"event_date": dates, "net_return": np.full(len(dates), 0.10)})
    result = monthly_block_bootstrap(trades, samples=200, seed=7)
    assert result["ci_low"] > 0
    assert result["probability_positive"] == 1.0


def test_cluster_accepted_rules_deduplicates_nested_candidates():
    events = pd.DataFrame(
        {
            "event_return": [0.50, 0.50, 0.50, 0.50],
            "close_location": [0.10, 0.15, 0.25, 0.70],
            "relative_dollar_volume": [5.0, 5.0, 5.0, 5.0],
            "event_close": [10.0, 10.0, 10.0, 10.0],
        }
    )
    validation_cfg = {
        "parameter_grid": {
            "continuation_return_bands": [[0.40, 0.60]],
            "continuation_close_locations": [0.90],
            "failed_spike_close_locations": [0.20, 0.30],
            "relative_volumes": [2],
            "minimum_prices": [3],
        }
    }
    accepted = pd.DataFrame(
        [
            {
                "rule": "failed_spike_cl0.20_rv2_px3",
                "side": "short",
                "horizon": 60,
                "positive_fold_fraction": 1.0,
                "oos_trimmed_mean_return": 0.30,
                "oos_n": 100,
                "oos_mean_return": 0.25,
                "oos_win_rate": 0.80,
                "fdr_q_value": 0.001,
            },
            {
                "rule": "failed_spike_cl0.30_rv2_px3",
                "side": "short",
                "horizon": 60,
                "positive_fold_fraction": 1.0,
                "oos_trimmed_mean_return": 0.28,
                "oos_n": 120,
                "oos_mean_return": 0.24,
                "oos_win_rate": 0.79,
                "fdr_q_value": 0.002,
            },
        ]
    )
    clusters = cluster_accepted_rules(
        events, accepted, validation_cfg, similarity_threshold=0.60
    )
    assert len(clusters) == 1
    assert clusters.iloc[0]["cluster_size"] == 2


def test_portfolio_respects_concurrent_position_limit():
    trades = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
            "exit_date": pd.to_datetime(["2020-02-01", "2020-02-02", "2020-02-03"]),
            "net_return": [0.10, 0.10, 0.10],
        }
    )
    ledger, summary = simulate_portfolio(
        trades,
        {
            "initial_capital": 100000,
            "position_fraction": 0.10,
            "max_concurrent_positions": 2,
            "max_daily_entries": 1,
            "stop_loss": None,
        },
    )
    assert len(ledger) == 2
    assert summary["trades"] == 2
    assert summary["ending_capital"] == pytest.approx(102000)


def test_stop_loss_caps_trade_loss():
    trades = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2020-01-01"]),
            "exit_date": pd.to_datetime(["2020-02-01"]),
            "net_return": [-0.80],
        }
    )
    ledger, _ = simulate_portfolio(
        trades,
        {
            "initial_capital": 100000,
            "position_fraction": 0.10,
            "max_concurrent_positions": 1,
            "max_daily_entries": 1,
            "stop_loss": 0.25,
        },
    )
    assert ledger.iloc[0]["return"] == pytest.approx(-0.25)
