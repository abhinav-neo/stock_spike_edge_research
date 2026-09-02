import numpy as np
import pandas as pd

from src.markov_regime_features import (
    add_markov_context,
    add_online_transition_probabilities,
    build_markov_features,
    classify_causal_regimes,
)


def benchmark(values):
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=len(values), freq="B"), "benchmark": values})


def test_regimes_are_invariant_to_future_prices():
    values = 100 * np.exp(np.cumsum(np.sin(np.arange(500) / 17) * 0.01))
    original = classify_causal_regimes(benchmark(values), minimum_history=40)
    changed = values.copy()
    changed[400:] *= np.linspace(1, 4, 100)
    revised = classify_causal_regimes(benchmark(changed), minimum_history=40)
    pd.testing.assert_frame_equal(original.iloc[:400], revised.iloc[:400])


def test_transition_probability_uses_only_observed_transitions():
    regimes = pd.DataFrame({"markov_state": pd.Series([0, 0, 1, 0], dtype="Int64")})
    result = add_online_transition_probabilities(regimes, state_count=2, prior=1.0)
    assert result.loc[0, "markov_next_quiet_bear_prob"] == 0.5
    assert result.loc[1, "markov_next_quiet_bear_prob"] == 2 / 3
    assert result.loc[2, "markov_next_quiet_bear_prob"] == 0.5
    assert result.loc[3, "markov_next_quiet_bear_prob"] == 0.5


def test_feature_builder_emits_probabilities_and_state_flags():
    values = 100 * np.exp(np.cumsum(np.sin(np.arange(400) / 13) * 0.02))
    result = build_markov_features(benchmark(values), minimum_history=30)
    ready = result.dropna(subset=["markov_state"])
    probability_columns = [column for column in result if column.startswith("markov_next_")]
    assert not ready.empty
    assert np.allclose(ready[probability_columns].sum(axis=1), 1.0)
    assert ready.filter(like="markov_is_").sum(axis=1).eq(1).all()


def test_join_retains_intentional_warmup_rows():
    events = pd.DataFrame({"symbol": ["A"], "event_date": ["2024-01-02"]})
    features = pd.DataFrame({"event_date": ["2024-01-02"], "markov_state": [pd.NA]})
    result = add_markov_context(events, features)
    assert len(result) == 1
    assert pd.isna(result.loc[0, "markov_state"])
