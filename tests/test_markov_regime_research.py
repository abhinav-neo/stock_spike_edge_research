import pandas as pd
import pytest

from src.markov_regime_research import evaluate_challengers


def model_frame(markov=False):
    rows = []
    for year in range(2015, 2024):
        for index in range(20):
            feature = index / 20
            row = {
                "symbol": f"S{index}",
                "event_date": pd.Timestamp(year, 1, 1) + pd.Timedelta(days=index),
                "event_return": feature,
                "forward_return_5d": feature * 0.1,
            }
            if markov:
                row["markov_state_persistence_prob"] = 0.8
            rows.append(row)
    return pd.DataFrame(rows)


def test_challenger_never_reads_locked_test_for_selection():
    baseline = model_frame()
    markov = model_frame(markov=True)
    _, original = evaluate_challengers(baseline, markov)
    baseline.loc[baseline.event_date.dt.year >= 2023, "forward_return_5d"] = -999
    markov.loc[markov.event_date.dt.year >= 2023, "forward_return_5d"] = 999
    _, changed = evaluate_challengers(baseline, markov)
    assert original["eligible_count"] == changed["eligible_count"]
    assert original["locked_candidate"] == changed["locked_candidate"]
    assert original["baseline_validation_correlation"] == pytest.approx(
        changed["baseline_validation_correlation"]
    )
    assert original["baseline_train_validation_gap"] == pytest.approx(
        changed["baseline_train_validation_gap"]
    )
