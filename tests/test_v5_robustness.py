import pandas as pd
import pytest

from src.ranked_portfolio_backtest import (
    apply_tradeability_filters,
    join_feature_data,
    score_threshold,
    simulate,
)
from src.train_predictive_model import chronological_split, select_features, target_horizon
from src.walk_forward_validation import prior_year_training_mask


def test_chronological_split_purges_labels_crossing_boundaries():
    data = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2019-12-20", "2019-12-30", "2022-12-20", "2022-12-29", "2023-01-03"]),
            "entry_date": pd.to_datetime(["2019-12-23", "2019-12-31", "2022-12-21", "2022-12-30", "2023-01-04"]),
        }
    )
    train, validation, test = chronological_split(
        data, "2019-12-31", "2022-12-31", target_horizon_days=5
    )
    assert train["event_date"].tolist() == [pd.Timestamp("2019-12-20")]
    assert validation["event_date"].tolist() == [pd.Timestamp("2022-12-20")]
    assert test["event_date"].tolist() == [pd.Timestamp("2023-01-03")]


def test_forward_outcomes_are_never_features():
    data = pd.DataFrame(
        {
            "event_return": [1.0],
            "entry_open": [10.0],
            "forward_return_5d": [2.0],
            "forward_return_20d": [3.0],
            "max_forward_60d_return": [4.0],
            "gain_retention_5d": [5.0],
            "above_entry_open_5d": [1],
            "days_to_5pct_breach": [2],
        }
    )
    assert select_features(data, "forward_return_5d") == ["event_return"]
    assert target_horizon("forward_return_5d") == 5


def test_walk_forward_training_uses_only_complete_prior_year_labels():
    data = pd.DataFrame(
        {
            "event_date": pd.to_datetime(["2023-12-20", "2023-12-29", "2024-01-02"]),
            "entry_date": pd.to_datetime(["2023-12-21", "2024-01-02", "2024-01-03"]),
        }
    )
    assert prior_year_training_mask(data, 2024, 5).tolist() == [True, False, False]


def test_tradeability_join_is_by_symbol_and_date_not_row_order():
    predictions = pd.DataFrame(
        {"symbol": ["AAA", "BBB"], "event_date": ["2024-01-02", "2024-01-03"]}
    )
    features = pd.DataFrame(
        {
            "symbol": ["BBB", "AAA"],
            "event_date": pd.to_datetime(["2024-01-03", "2024-01-02"]),
            "event_close": [20.0, 5.0],
            "prior_20d_avg_dollar_volume": [2_000_000.0, 100_000.0],
        }
    )
    joined = join_feature_data(predictions, features)
    assert joined["event_close"].tolist() == [5.0, 20.0]
    filtered, diagnostics = apply_tradeability_filters(joined, 10.0, 1_000_000.0, None)
    assert filtered["symbol"].tolist() == ["BBB"]
    assert diagnostics["min_price_column"] == "event_close"
    assert diagnostics["min_dollar_volume_column"] == "prior_20d_avg_dollar_volume"


def test_feature_join_rejects_duplicate_keys():
    predictions = pd.DataFrame({"symbol": ["AAA"], "event_date": ["2024-01-02"]})
    features = pd.DataFrame(
        {"symbol": ["AAA", "AAA"], "event_date": ["2024-01-02", "2024-01-02"]}
    )
    with pytest.raises(ValueError, match="duplicate"):
        join_feature_data(predictions, features)


def test_threshold_uses_validation_only_and_costs_are_applied_once():
    source = pd.DataFrame(
        {
            "period": ["validation", "validation", "test"],
            "predicted_return": [-0.20, -0.10, -99.0],
        }
    )
    threshold = score_threshold(source, "short", 0.5, "validation")
    assert threshold == pytest.approx(-0.15)

    evaluation = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "event_date": pd.to_datetime(["2024-01-02"]),
            "predicted_return": [-1.0],
            "actual_return": [0.10],
        }
    )
    trades, _, _ = simulate(
        evaluation, "short", 0.0, 0.1, "validation", 5,
        100_000.0, 10, 30.0, 10.0, True,
    )
    # Short gross return is -10%; subtract 30 bps round-trip and 5*10 bps borrow once.
    assert trades.loc[0, "net_strategy_return"] == pytest.approx(-0.108)


def test_same_symbol_overlap_is_prevented_and_sizing_is_fixed():
    evaluation = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "event_date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-03"]),
            "predicted_return": [-3.0, -2.0, -1.0],
            "actual_return": [-0.10, -0.10, -0.10],
        }
    )
    trades, _, summary = simulate(
        evaluation, "short", 0.0, 0.1, "validation", 5,
        100_000.0, 10, 0.0, 0.0, True,
    )
    assert trades["symbol"].tolist() == ["AAA", "BBB"]
    assert trades["notional"].tolist() == [10_000.0, 10_000.0]
    assert summary["skipped_due_to_symbol_overlap"] == 1


def test_simulation_uses_next_session_entry_date_when_available():
    evaluation = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "event_date": pd.to_datetime(["2024-01-02"]),
            "entry_date": pd.to_datetime(["2024-01-03"]),
            "predicted_return": [-1.0],
            "actual_return": [-0.10],
        }
    )
    trades, _, _ = simulate(
        evaluation, "short", 0.0, 0.1, "validation", 5,
        100_000.0, 10, 0.0, 0.0, True,
    )
    assert trades.loc[0, "entry_date"] == pd.Timestamp("2024-01-03")
