import numpy as np
import pandas as pd

from src.walk_forward import (
    benjamini_hochberg,
    build_folds,
    evaluate_walk_forward,
    trimmed_mean,
)


def test_build_folds_uses_expanding_training_windows():
    df = pd.DataFrame({"event_date": pd.date_range("2015-01-01", "2024-12-31", freq="180D")})
    folds = build_folds(
        df,
        {"oos_start": "2020-01-01", "test_years": 1, "minimum_train_years": 5},
    )
    assert folds[0]["train_end"] == pd.Timestamp("2019-12-31")
    assert folds[0]["test_start"] == pd.Timestamp("2020-01-01")
    assert folds[1]["train_end"] == pd.Timestamp("2020-12-31")
    assert folds[1]["test_start"] == pd.Timestamp("2021-01-01")


def test_benjamini_hochberg_is_monotone_and_bounded():
    adjusted = benjamini_hochberg(pd.Series([0.001, 0.01, 0.04, 0.50]))
    assert adjusted.between(0, 1).all()
    assert adjusted.iloc[0] <= adjusted.iloc[1] <= adjusted.iloc[2] <= adjusted.iloc[3]


def test_trimmed_mean_reduces_outlier_influence():
    values = pd.Series([0.10] * 9 + [10.0])
    assert trimmed_mean(values, 0.10) < values.mean()


def test_walk_forward_never_marks_production_approved():
    dates = pd.date_range("2015-01-02", "2024-12-20", freq="14D")
    n = len(dates)
    df = pd.DataFrame(
        {
            "event_date": dates,
            "event_return": np.full(n, 0.50),
            "close_location": np.full(n, 0.10),
            "relative_dollar_volume": np.full(n, 5.0),
            "event_close": np.full(n, 10.0),
            "forward_return_20d": np.full(n, -0.20),
        }
    )
    validation_cfg = {
        "transaction_cost_bps_each_way": 0,
        "slippage_bps_each_way": 0,
        "horizons": [20],
        "parameter_grid": {
            "continuation_return_bands": [[0.40, 0.60]],
            "continuation_close_locations": [0.90],
            "failed_spike_close_locations": [0.20],
            "relative_volumes": [2],
            "minimum_prices": [3],
        },
    }
    wf_cfg = {
        "oos_start": "2020-01-01",
        "test_years": 1,
        "minimum_train_years": 5,
        "minimum_train_events_per_fold": 20,
        "minimum_combined_oos_events": 50,
        "minimum_positive_fold_fraction": 0.60,
        "fdr_alpha": 0.05,
        "short_borrow_bps_annual": 0,
    }
    summary, details = evaluate_walk_forward(df, validation_cfg, wf_cfg)
    assert not details.empty
    assert not summary.empty
    assert set(summary["research_status"]).issubset({"research_candidate", "rejected"})
    assert "production_approved" not in set(summary["research_status"])
