from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analyze_edges import candidate_masks, performance_stats


def build_folds(df: pd.DataFrame, cfg: dict) -> list[dict]:
    dates = pd.to_datetime(df["event_date"])
    start = pd.Timestamp(cfg["oos_start"])
    end = dates.max().normalize()
    test_years = int(cfg.get("test_years", 1))
    minimum_train_years = int(cfg.get("minimum_train_years", 5))
    folds: list[dict] = []
    fold_start = start
    fold_number = 1
    while fold_start <= end:
        fold_end = min(fold_start + pd.DateOffset(years=test_years) - pd.Timedelta(days=1), end)
        train_end = fold_start - pd.Timedelta(days=1)
        train_start = dates.min()
        if train_end >= train_start + pd.DateOffset(years=minimum_train_years):
            folds.append({
                "fold": fold_number,
                "train_start": train_start,
                "train_end": train_end,
                "test_start": fold_start,
                "test_end": fold_end,
            })
            fold_number += 1
        fold_start = fold_start + pd.DateOffset(years=test_years)
    if not folds:
        raise ValueError("walk_forward configuration produced no valid folds")
    return folds


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    p = p_values.fillna(1.0).clip(0.0, 1.0).to_numpy(float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(n)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(result, index=p_values.index)


def normal_two_sided_p_value(t_stat: float) -> float:
    if not np.isfinite(t_stat):
        return 1.0
    return float(np.math.erfc(abs(t_stat) / np.sqrt(2.0)))


def trimmed_mean(values: pd.Series, proportion: float = 0.10) -> float:
    values = values.dropna().sort_values()
    if values.empty:
        return np.nan
    trim = int(len(values) * proportion)
    if trim == 0 or 2 * trim >= len(values):
        return float(values.mean())
    return float(values.iloc[trim:-trim].mean())


def evaluate_walk_forward(df: pd.DataFrame, validation_cfg: dict, wf_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df.copy()
    work["event_date"] = pd.to_datetime(work["event_date"])
    folds = build_folds(work, wf_cfg)
    cost = 2 * (
        validation_cfg["transaction_cost_bps_each_way"]
        + validation_cfg["slippage_bps_each_way"]
    ) / 10_000
    short_borrow_bps_annual = float(wf_cfg.get("short_borrow_bps_annual", 0.0))
    horizons = validation_cfg.get("horizons", [1, 2, 3, 5, 10, 20, 40, 60])
    minimum_train = int(wf_cfg.get("minimum_train_events_per_fold", 30))
    minimum_oos = int(wf_cfg.get("minimum_combined_oos_events", 100))
    minimum_positive_fraction = float(wf_cfg.get("minimum_positive_fold_fraction", 0.60))
    fdr_alpha = float(wf_cfg.get("fdr_alpha", 0.05))

    fold_rows: list[dict] = []
    summary_rows: list[dict] = []
    for candidate in candidate_masks(work, validation_cfg):
        for horizon in horizons:
            col = f"forward_return_{horizon}d"
            if col not in work:
                continue
            oos_parts: list[pd.Series] = []
            positive_folds = 0
            eligible_folds = 0
            for fold in folds:
                train_mask = work["event_date"].between(fold["train_start"], fold["train_end"])
                test_mask = work["event_date"].between(fold["test_start"], fold["test_end"])
                train_returns = work.loc[candidate["mask"] & train_mask, col]
                test_returns = work.loc[candidate["mask"] & test_mask, col]
                if candidate["side"] == "short":
                    train_returns = -train_returns
                    test_returns = -test_returns
                    borrow_cost = short_borrow_bps_annual / 10_000 * horizon / 252
                else:
                    borrow_cost = 0.0
                train_stats = performance_stats(train_returns, cost + borrow_cost)
                test_net = test_returns.dropna() - cost - borrow_cost
                test_stats = performance_stats(test_returns, cost + borrow_cost)
                eligible = train_stats["n"] >= minimum_train
                if eligible:
                    eligible_folds += 1
                    oos_parts.append(test_net)
                    positive_folds += int(test_stats["mean_return"] > 0)
                fold_rows.append({
                    "rule": candidate["rule"], "horizon": horizon, "side": candidate["side"],
                    "fold": fold["fold"], "train_end": fold["train_end"],
                    "test_start": fold["test_start"], "test_end": fold["test_end"],
                    "eligible": eligible, "train_n": train_stats["n"],
                    "train_mean_return": train_stats["mean_return"],
                    "oos_n": test_stats["n"], "oos_mean_return": test_stats["mean_return"],
                    "oos_median_return": test_stats["median_return"],
                    "oos_win_rate": test_stats["win_rate"], "oos_t_stat": test_stats["t_stat"],
                })
            combined = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.Series(dtype=float)
            stats = performance_stats(combined, 0.0)
            positive_fraction = positive_folds / eligible_folds if eligible_folds else 0.0
            row = {key: value for key, value in candidate.items() if key != "mask"}
            row.update({
                "horizon": horizon, "eligible_folds": eligible_folds,
                "positive_folds": positive_folds, "positive_fold_fraction": positive_fraction,
                "oos_n": stats["n"], "oos_mean_return": stats["mean_return"],
                "oos_median_return": stats["median_return"],
                "oos_trimmed_mean_return": trimmed_mean(combined),
                "oos_win_rate": stats["win_rate"], "oos_t_stat": stats["t_stat"],
                "raw_p_value": normal_two_sided_p_value(stats["t_stat"]),
                "sample_size_pass": stats["n"] >= minimum_oos,
                "fold_consistency_pass": positive_fraction >= minimum_positive_fraction,
            })
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    details = pd.DataFrame(fold_rows)
    if not summary.empty:
        summary["fdr_q_value"] = benjamini_hochberg(summary["raw_p_value"])
        summary["multiple_testing_pass"] = summary["fdr_q_value"] <= fdr_alpha
        summary["accepted"] = (
            summary["sample_size_pass"]
            & summary["fold_consistency_pass"]
            & summary["multiple_testing_pass"]
            & (summary["oos_mean_return"] > 0)
            & (summary["oos_median_return"] > 0)
        )
        summary["research_status"] = np.where(summary["accepted"], "research_candidate", "rejected")
        summary = summary.sort_values(
            ["accepted", "positive_fold_fraction", "oos_trimmed_mean_return"],
            ascending=[False, False, False],
        )
    return summary, details


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--events", default="data/processed/events.parquet")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    summary, folds = evaluate_walk_forward(
        pd.read_parquet(args.events), config["validation"], config["walk_forward"]
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "walk_forward_summary.csv", index=False)
    folds.to_csv(out / "walk_forward_folds.csv", index=False)
    accepted = summary[summary["accepted"].fillna(False)]
    accepted.to_csv(out / "walk_forward_accepted.csv", index=False)
    print("\nTop walk-forward candidates")
    columns = ["rule", "horizon", "side", "eligible_folds", "positive_fold_fraction", "oos_n", "oos_mean_return", "oos_median_return", "oos_trimmed_mean_return", "oos_win_rate", "oos_t_stat", "fdr_q_value", "accepted", "research_status"]
    print(summary[[c for c in columns if c in summary]].head(20).to_string(index=False))
    print(f"\nWalk-forward candidates evaluated: {len(summary)}")
    print(f"Accepted research candidates: {len(accepted)}")
    print("Production-approved candidates: 0")


if __name__ == "__main__":
    main()
