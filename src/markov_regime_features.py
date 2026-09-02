from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.v5_mtm_research import normalize_spy


STATE_NAMES = ("quiet_bear", "neutral", "quiet_bull", "stress")


def _past_percentile(values: pd.Series, minimum_history: int) -> pd.Series:
    """Percentile of today's value against strictly prior observations."""
    numeric = pd.to_numeric(values, errors="coerce")
    output = pd.Series(np.nan, index=numeric.index, dtype=float)
    history: list[float] = []
    for index, value in numeric.items():
        if np.isfinite(value) and len(history) >= minimum_history:
            prior = np.asarray(history, dtype=float)
            output.loc[index] = (np.count_nonzero(prior < value) + 0.5 * np.count_nonzero(prior == value)) / len(prior)
        if np.isfinite(value):
            history.append(float(value))
    return output


def classify_causal_regimes(
    spy: pd.DataFrame,
    minimum_history: int = 252,
) -> pd.DataFrame:
    frame = spy.copy().sort_values("date").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    close = pd.to_numeric(frame["benchmark"], errors="coerce")
    returns = close.pct_change()
    momentum = close.pct_change(20)
    volatility = returns.rolling(20, min_periods=20).std()
    momentum_pct = _past_percentile(momentum, minimum_history)
    volatility_pct = _past_percentile(volatility, minimum_history)

    state = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    ready = momentum_pct.notna() & volatility_pct.notna()
    state.loc[ready] = 1
    state.loc[ready & (momentum_pct >= 0.60) & (volatility_pct < 0.80)] = 2
    state.loc[ready & (momentum_pct <= 0.20) & (volatility_pct < 0.80)] = 0
    state.loc[ready & (volatility_pct >= 0.80)] = 3
    return pd.DataFrame(
        {
            "date": frame["date"],
            "markov_momentum_20d": momentum,
            "markov_volatility_20d": volatility,
            "markov_momentum_percentile": momentum_pct,
            "markov_volatility_percentile": volatility_pct,
            "markov_state": state,
        }
    )


def add_online_transition_probabilities(
    regimes: pd.DataFrame,
    state_count: int = 4,
    prior: float = 1.0,
) -> pd.DataFrame:
    """Estimate P(S[t+1] | S[t]) using transitions observed through t only."""
    result = regimes.copy().reset_index(drop=True)
    counts = np.full((state_count, state_count), float(prior))
    probabilities = np.full((len(result), state_count), np.nan)
    previous: int | None = None
    for position, raw_state in enumerate(result["markov_state"]):
        if pd.isna(raw_state):
            continue
        current = int(raw_state)
        if previous is not None:
            counts[previous, current] += 1.0
        probabilities[position] = counts[current] / counts[current].sum()
        previous = current
    for state, name in enumerate(STATE_NAMES[:state_count]):
        result[f"markov_next_{name}_prob"] = probabilities[:, state]
    result["markov_state_persistence_prob"] = np.where(
        result["markov_state"].notna(),
        probabilities[np.arange(len(result)), result["markov_state"].fillna(0).astype(int)],
        np.nan,
    )
    return result


def build_markov_features(spy: pd.DataFrame, minimum_history: int = 252) -> pd.DataFrame:
    regimes = add_online_transition_probabilities(classify_causal_regimes(spy, minimum_history))
    regimes = regimes.rename(columns={"date": "event_date"})
    for state, name in enumerate(STATE_NAMES):
        regimes[f"markov_is_{name}"] = (regimes["markov_state"] == state).astype("Int64")
    return regimes


def add_markov_context(events: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Join regimes while retaining the intentional causal warm-up rows."""
    left = events.copy()
    right = features.copy()
    left["event_date"] = pd.to_datetime(left["event_date"]).dt.normalize()
    right["event_date"] = pd.to_datetime(right["event_date"]).dt.normalize()
    if right.duplicated("event_date").any():
        raise ValueError("Markov feature data contains duplicate event dates")
    missing_dates = set(left["event_date"]) - set(right["event_date"])
    if missing_dates:
        examples = sorted(date.strftime("%Y-%m-%d") for date in missing_dates)[:10]
        raise ValueError(f"Missing benchmark dates for events: {examples}")
    return left.merge(right, on="event_date", how="left", validate="many_to_one")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add causal Markov-chain market regimes to event features")
    parser.add_argument("--events", default="data/processed/events_features_v5.parquet")
    parser.add_argument("--spy", default="data/raw/spy_benchmark.parquet")
    parser.add_argument("--output", default="data/processed/events_features_v7_markov.parquet")
    parser.add_argument("--minimum-history", type=int, default=252)
    args = parser.parse_args()

    events = pd.read_parquet(args.events)
    features = build_markov_features(normalize_spy(Path(args.spy)), args.minimum_history)
    enriched = add_markov_context(events, features)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output, index=False)
    enriched.to_csv(output.with_suffix(".csv"), index=False)
    print(f"Wrote {len(enriched):,} events with causal Markov regimes to {output}")


if __name__ == "__main__":
    main()
