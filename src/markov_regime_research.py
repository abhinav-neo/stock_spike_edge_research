from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.free_data_factorial_research import evaluate_variant


def evaluate_challengers(
    baseline: pd.DataFrame,
    markov: pd.DataFrame,
    target: str = "forward_return_5d",
    minimum_improvement: float = 0.02,
    maximum_excess_gap: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    for name, frame in (("baseline", baseline), ("markov", markov)):
        for model in ("random_forest", "hist_gradient_boosting"):
            row, _ = evaluate_variant(frame, (name,), model, target)
            row["variant"] = name
            rows.append(row)
    results = pd.DataFrame(rows).sort_values("validation_correlation", ascending=False).reset_index(drop=True)
    baseline_row = results.loc[
        (results["variant"] == "baseline") & (results["model"] == "random_forest")
    ].iloc[0]
    challengers = results.loc[results["variant"] == "markov"].copy()
    challengers["validation_improvement"] = (
        challengers["validation_correlation"] - baseline_row["validation_correlation"]
    )
    challengers["excess_gap"] = challengers["train_validation_gap"] - baseline_row["train_validation_gap"]
    challengers["eligible"] = (
        (challengers["validation_improvement"] >= minimum_improvement)
        & (challengers["excess_gap"] <= maximum_excess_gap)
    )
    locked = challengers.loc[challengers["eligible"]].sort_values("validation_correlation", ascending=False)
    summary = {
        "protocol": "Validation-only Markov challenger; locked test is not evaluated unless a challenger passes both gates.",
        "target": target,
        "baseline_validation_correlation": float(baseline_row["validation_correlation"]),
        "baseline_train_validation_gap": float(baseline_row["train_validation_gap"]),
        "minimum_validation_improvement": minimum_improvement,
        "maximum_excess_train_validation_gap": maximum_excess_gap,
        "eligible_count": int(challengers["eligible"].sum()),
        "locked_candidate": locked.iloc[0].to_dict() if not locked.empty else None,
    }
    return results.merge(
        challengers[["variant", "model", "validation_improvement", "excess_gap", "eligible"]],
        on=["variant", "model"],
        how="left",
    ), summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only evaluation of causal Markov regime features")
    parser.add_argument("--baseline", default="data/processed/events_features_v5.parquet")
    parser.add_argument("--markov", default="data/processed/events_features_v7_markov.parquet")
    parser.add_argument("--target", default="forward_return_5d")
    parser.add_argument("--minimum-improvement", type=float, default=0.02)
    parser.add_argument("--maximum-excess-gap", type=float, default=0.05)
    parser.add_argument("--output-dir", default="reports/markov_regime")
    args = parser.parse_args()

    results, summary = evaluate_challengers(
        pd.read_parquet(args.baseline),
        pd.read_parquet(args.markov),
        args.target,
        args.minimum_improvement,
        args.maximum_excess_gap,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "validation_metrics.csv", index=False)
    (output / "selection_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(results.to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
