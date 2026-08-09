from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.intraday_execution import ExecutionCosts, account_metrics, simulate_round_trips, validate_quotes


def online_state_persistence(states: pd.Series, state_count: int = 3, prior: float = 1.0) -> np.ndarray:
    counts = np.full((state_count, state_count), prior, dtype=float)
    result = np.empty(len(states), dtype=float)
    previous = None
    for position, state in enumerate(states.astype(int)):
        if previous is not None:
            counts[previous, state] += 1
        result[position] = counts[state, state] / counts[state].sum()
        previous = state
    return result


def build_imbalance_signals(
    quotes: pd.DataFrame,
    threshold: float,
    minimum_persistence: float,
    holding_seconds: float,
    capital: float = 10_000.0,
    position_fraction: float = 0.10,
) -> pd.DataFrame:
    data = validate_quotes(quotes)
    denominator = data["bid_size"] + data["ask_size"]
    imbalance = (data["bid_size"] - data["ask_size"]) / denominator.replace(0, np.nan)
    states = pd.Series(np.where(imbalance > threshold, 2, np.where(imbalance < -threshold, 0, 1)))
    persistence = online_state_persistence(states)
    rows = []
    next_available = data["timestamp"].iloc[0]
    for position in range(len(data)):
        timestamp = data["timestamp"].iloc[position]
        if timestamp < next_available or not np.isfinite(imbalance.iloc[position]):
            continue
        if abs(float(imbalance.iloc[position])) < threshold or persistence[position] < minimum_persistence:
            continue
        midpoint = (float(data["bid"].iloc[position]) + float(data["ask"].iloc[position])) / 2
        quantity = np.floor(capital * position_fraction / midpoint)
        if quantity < 1:
            continue
        exit_timestamp = timestamp + pd.Timedelta(seconds=holding_seconds)
        rows.append(
            {
                "symbol": data["symbol"].iloc[position],
                "signal_timestamp": timestamp,
                "exit_timestamp": exit_timestamp,
                "side": "buy" if imbalance.iloc[position] > 0 else "sell",
                "quantity": quantity,
                "imbalance": float(imbalance.iloc[position]),
                "state_persistence": float(persistence[position]),
            }
        )
        next_available = exit_timestamp
    return pd.DataFrame(rows)


def session_metrics(fills: pd.DataFrame, initial_capital: float = 10_000.0) -> dict:
    _, summary = account_metrics(fills, initial_capital)
    if fills.empty:
        return {**summary, "net_basis_points": 0.0, "profit_factor": np.nan}
    wins = fills.loc[fills["net_pnl"] > 0, "net_pnl"].sum()
    losses = -fills.loc[fills["net_pnl"] < 0, "net_pnl"].sum()
    return {
        **summary,
        "net_basis_points": float(summary["net_pnl"] / initial_capital * 10_000),
        "profit_factor": float(wins / losses) if losses > 0 else np.nan,
    }


def run_study(quotes: pd.DataFrame, costs: ExecutionCosts) -> tuple[pd.DataFrame, dict]:
    data = validate_quotes(quotes)
    start, end = data["timestamp"].iloc[0], data["timestamp"].iloc[-1]
    train_end = start + (end - start) / 2
    validation_end = start + (end - start) * 0.75
    candidates = list(itertools.product((0.20, 0.40, 0.60), (0.40, 0.60, 0.80), (1, 5, 30)))
    rows = []
    cached: dict[int, pd.DataFrame] = {}
    for candidate_id, (threshold, persistence, holding) in enumerate(candidates):
        signals = build_imbalance_signals(data, threshold, persistence, holding)
        fills, rejected = simulate_round_trips(data, signals, costs)
        cached[candidate_id] = fills
        train_fills = fills.loc[(fills["entry_timestamp"] < train_end) & (fills["exit_timestamp"] < train_end)]
        metrics = session_metrics(train_fills)
        rows.append(
            {
                "candidate_id": candidate_id,
                "threshold": threshold,
                "minimum_persistence": persistence,
                "holding_seconds": holding,
                "signals": len(signals),
                "rejections": len(rejected),
                **{f"train_{key}": value for key, value in metrics.items()},
            }
        )
    results = pd.DataFrame(rows)
    shortlist = results.loc[results["train_trades"] >= 20].sort_values(
        ["train_net_pnl", "train_max_drawdown"], ascending=[False, False]
    ).head(5)
    validation_rows = []
    for candidate_id in shortlist["candidate_id"].astype(int):
        fills = cached[candidate_id]
        selected = fills.loc[
            (fills["entry_timestamp"] >= train_end)
            & (fills["entry_timestamp"] < validation_end)
            & (fills["exit_timestamp"] < validation_end)
        ]
        metrics = session_metrics(selected)
        validation_rows.append({"candidate_id": candidate_id, **{f"validation_{key}": value for key, value in metrics.items()}})
    results = results.merge(pd.DataFrame(validation_rows), on="candidate_id", how="left")
    eligible = results.loc[
        (results["validation_net_pnl"] > 0)
        & (results["validation_profit_factor"] >= 1.25)
        & (results["validation_trades"] >= 10)
    ].sort_values("validation_net_pnl", ascending=False)
    summary = {
        "status": "single_2003_session_parser_and_execution_validation_only",
        "source_quotes": int(len(data)),
        "candidate_count": len(candidates),
        "shortlist_count": int(len(shortlist)),
        "validation_positive_count": int((results["validation_net_pnl"] > 0).sum()),
        "eligible_count": int(len(eligible)),
        "locked_test_evaluated": False,
        "reason_test_withheld": "One public session cannot support model selection or annual-return inference.",
    }
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Causal order-imbalance study on an extracted ITCH quote sample")
    parser.add_argument("--quotes", default="data/processed/nasdaq_itch2_msft_quotes.parquet")
    parser.add_argument("--output-dir", default="reports/itch_imbalance_sample")
    args = parser.parse_args()
    costs = ExecutionCosts(
        latency_ms=100,
        commission_per_share=0.005,
        minimum_commission=0.35,
        impact_bps_at_full_touch=2.0,
        maximum_participation=0.10,
    )
    results, summary = run_study(pd.read_parquet(args.quotes), costs)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "candidate_metrics.csv", index=False)
    (output / "selection_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(results.sort_values("validation_net_pnl", ascending=False).head(10).to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
