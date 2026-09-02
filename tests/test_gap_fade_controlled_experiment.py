from __future__ import annotations

import pandas as pd

from src.gap_fade_controlled_experiment import (
    apply_trade_quality_filters,
    select_gap_fade_short,
)


def test_select_gap_fade_short_excludes_other_families() -> None:
    survivors = pd.DataFrame([
        {
            "family": "gap_fade",
            "direction": "short",
            "horizon": 5,
            "test_cluster_t_stat": 3.0,
            "test_daily_mean": 0.02,
        },
        {
            "family": "mean_reversion",
            "direction": "short",
            "horizon": 10,
            "test_cluster_t_stat": 4.0,
            "test_daily_mean": 0.03,
        },
        {
            "family": "gap_fade",
            "direction": "long",
            "horizon": 5,
            "test_cluster_t_stat": 5.0,
            "test_daily_mean": 0.04,
        },
    ])
    selected = select_gap_fade_short(survivors, maximum_candidates=2)
    assert len(selected) == 1
    assert selected.iloc[0]["family"] == "gap_fade"
    assert selected.iloc[0]["direction"] == "short"


def test_quality_filters_reject_bad_prices_and_low_liquidity() -> None:
    trades = pd.DataFrame([
        {
            "symbol": "GOOD",
            "entry_price": 20.0,
            "avg_dollar_volume_20d": 25_000_000.0,
            "signal_close_ratio": 1.20,
        },
        {
            "symbol": "PENNY",
            "entry_price": 4.0,
            "avg_dollar_volume_20d": 25_000_000.0,
            "signal_close_ratio": 1.10,
        },
        {
            "symbol": "SPLIT",
            "entry_price": 28_500.0,
            "avg_dollar_volume_20d": 50_000_000.0,
            "signal_close_ratio": 10.0,
        },
        {
            "symbol": "ILLIQUID",
            "entry_price": 12.0,
            "avg_dollar_volume_20d": 2_000_000.0,
            "signal_close_ratio": 1.05,
        },
    ])

    accepted, rejected = apply_trade_quality_filters(trades)

    assert accepted["symbol"].tolist() == ["GOOD"]
    assert set(rejected["symbol"]) == {"PENNY", "SPLIT", "ILLIQUID"}
    split_reason = rejected.loc[rejected["symbol"].eq("SPLIT"), "rejection_reason"].iloc[0]
    assert "entry_price_above_maximum" in split_reason
    assert "possible_corporate_action_or_bad_price" in split_reason
