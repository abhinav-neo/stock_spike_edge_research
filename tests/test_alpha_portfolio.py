from __future__ import annotations

import pandas as pd

from src.alpha_portfolio import apply_capacity, select_representatives


def test_select_representatives_collapses_parameter_clones() -> None:
    survivors = pd.DataFrame([
        {"family": "gap_fade", "direction": "short", "horizon": 5, "test_cluster_t_stat": 4.0, "test_daily_mean": 0.03},
        {"family": "gap_fade", "direction": "short", "horizon": 5, "test_cluster_t_stat": 3.0, "test_daily_mean": 0.02},
        {"family": "gap_fade", "direction": "short", "horizon": 10, "test_cluster_t_stat": 2.5, "test_daily_mean": 0.02},
    ])
    selected = select_representatives(survivors, maximum=10, per_family=2)
    assert len(selected) == 2
    assert set(selected["horizon"]) == {5, 10}


def test_apply_capacity_enforces_daily_and_concurrent_limits() -> None:
    trades = pd.DataFrame([
        {"entry_date": "2024-01-02", "scheduled_exit_date": "2024-01-10", "candidate_rank": 1, "symbol": "AAA"},
        {"entry_date": "2024-01-02", "scheduled_exit_date": "2024-01-10", "candidate_rank": 2, "symbol": "BBB"},
        {"entry_date": "2024-01-03", "scheduled_exit_date": "2024-01-08", "candidate_rank": 1, "symbol": "CCC"},
    ])
    accepted, rejected = apply_capacity(trades, max_daily_entries=1, max_concurrent=1)
    assert len(accepted) == 1
    assert len(rejected) == 2
    assert set(rejected["rejection_reason"]) == {"daily_entry_cap", "concurrency_cap"}
