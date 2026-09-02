from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.train_predictive_model import build_pipeline, chronological_split, select_features, target_horizon


KEYS = ["symbol", "event_date"]
GROUP_PREFIXES = {
    "spy": ("spy_",),
    "vix": ("vix_",),
    "sector": ("sector_", "stock_minus_sector_", "prior_60d_sector_", "inferred_sector_"),
    "finra": ("finra_",),
    "ftd": ("ftd_",),
}


def feature_group(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    prefixes = GROUP_PREFIXES[name]
    columns = [column for column in frame.columns if column.startswith(prefixes)]
    if not columns:
        raise ValueError(f"No {name} columns found")
    result = frame[[*KEYS, *columns]].copy()
    if result.duplicated(KEYS).any():
        raise ValueError(f"Duplicate keys in {name} feature group")
    return result


def all_combinations(names: list[str]) -> list[tuple[str, ...]]:
    return [combo for size in range(len(names) + 1) for combo in itertools.combinations(names, size)]


def build_variant(base: pd.DataFrame, groups: dict[str, pd.DataFrame], combo: tuple[str, ...]) -> pd.DataFrame:
    result = base.copy()
    for name in combo:
        result = result.merge(groups[name], on=KEYS, how="left", validate="one_to_one")
    return result


def correlation(actual: pd.Series, predicted: np.ndarray) -> float:
    a, p = actual.to_numpy(float), np.asarray(predicted, dtype=float)
    valid = np.isfinite(a) & np.isfinite(p)
    return float(np.corrcoef(a[valid], p[valid])[0, 1]) if valid.sum() > 1 and np.std(a[valid]) > 0 and np.std(p[valid]) > 0 else np.nan


def evaluate_variant(frame: pd.DataFrame, combo: tuple[str, ...], model_name: str, target: str) -> tuple[dict, object]:
    data = frame.loc[frame[target].notna()].copy()
    data["event_date"] = pd.to_datetime(data["event_date"])
    train, validation, _ = chronological_split(data, "2019-12-31", "2022-12-31", target_horizon(target))
    features = select_features(data, target)
    pipeline = build_pipeline(train[features], model_name, 42)
    pipeline.fit(train[features], train[target])
    train_corr = correlation(train[target], pipeline.predict(train[features]))
    validation_corr = correlation(validation[target], pipeline.predict(validation[features]))
    return {
        "combination": "+".join(combo) if combo else "baseline",
        "group_count": len(combo), "model": model_name, "feature_count": len(features),
        "train_correlation": train_corr, "validation_correlation": validation_corr,
        "train_validation_gap": train_corr - validation_corr,
        "train_n": len(train), "validation_n": len(validation),
    }, pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only factorial study of all free-data feature groups")
    parser.add_argument("--base", default="data/processed/events_features_v5.parquet")
    parser.add_argument("--market", default="data/processed/events_features_v6_vix.parquet")
    parser.add_argument("--sector", default="data/processed/events_features_v6_sector.parquet")
    parser.add_argument("--finra", default="data/processed/events_features_v5_finra_ratio.parquet")
    parser.add_argument("--ftd", default="data/processed/events_features_v5_sec_ftd.parquet")
    parser.add_argument("--target", default="forward_return_5d")
    parser.add_argument("--min-improvement", type=float, default=0.02)
    parser.add_argument("--max-excess-train-validation-gap", type=float, default=0.05)
    parser.add_argument("--output-dir", default="reports/free_data_factorial")
    args = parser.parse_args()

    base = pd.read_parquet(args.base)
    market = pd.read_parquet(args.market)
    sources = {
        "spy": market, "vix": market, "sector": pd.read_parquet(args.sector),
        "finra": pd.read_parquet(args.finra), "ftd": pd.read_parquet(args.ftd),
    }
    groups = {name: feature_group(source, name) for name, source in sources.items()}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for combo in all_combinations(list(GROUP_PREFIXES)):
        variant = build_variant(base, groups, combo)
        for model in ("random_forest", "hist_gradient_boosting"):
            row, _ = evaluate_variant(variant, combo, model, args.target)
            rows.append(row)
            print(f"{row['combination']:40s} {model:24s} validation={row['validation_correlation']:.4f}")
    results = pd.DataFrame(rows).sort_values("validation_correlation", ascending=False).reset_index(drop=True)
    results.to_csv(output / "validation_factorial.csv", index=False)
    baseline = float(results.loc[(results["combination"] == "baseline") & (results["model"] == "random_forest"), "validation_correlation"].iloc[0])
    baseline_gap = float(results.loc[(results["combination"] == "baseline") & (results["model"] == "random_forest"), "train_validation_gap"].iloc[0])
    eligible = results.loc[
        (results["validation_correlation"] >= baseline + args.min_improvement)
        & (results["train_validation_gap"] <= baseline_gap + args.max_excess_train_validation_gap)
    ]
    locked = eligible.iloc[0].to_dict() if not eligible.empty else None
    summary = {
        "protocol": "All ranking and locking uses validation only; test is not evaluated by this module.",
        "groups": list(GROUP_PREFIXES), "combinations": 32, "model_variants": 2, "evaluations": 64,
        "baseline_validation_correlation": baseline, "minimum_absolute_improvement": args.min_improvement,
        "baseline_train_validation_gap": baseline_gap,
        "maximum_excess_train_validation_gap": args.max_excess_train_validation_gap,
        "eligible_count": int(len(eligible)), "locked_candidate": locked,
    }
    (output / "selection_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
