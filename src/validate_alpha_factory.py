from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.alpha_factory import build_features, candidate_mask, candidate_specs


def executable_forward_return(frame: pd.DataFrame, horizon: int, direction: str) -> pd.Series:
    group = frame.groupby("symbol", group_keys=False)
    entry_open = group["open"].shift(-1)
    exit_close = group["close"].shift(-int(horizon))
    gross = exit_close / entry_open - 1.0
    return gross if direction == "long" else -gross


def period_sample(frame: pd.DataFrame, spec: dict, cfg: dict, start: str | None, end: str | None) -> pd.DataFrame:
    mask = candidate_mask(frame, spec, cfg)
    dates = pd.to_datetime(frame["date"])
    if start:
        mask &= dates.ge(pd.Timestamp(start))
    if end:
        cutoff = pd.Timestamp(end) - pd.tseries.offsets.BDay(int(spec["horizon"]) + 1)
        mask &= dates.le(cutoff)
    forward = executable_forward_return(frame, int(spec["horizon"]), spec["direction"])
    cost = float(cfg.get("round_trip_cost_bps", 100.0)) / 10000.0
    sample = pd.DataFrame({"date": dates.loc[mask], "net": forward.loc[mask] - cost}).dropna()
    return sample


def daily_cluster_statistics(sample: pd.DataFrame) -> dict:
    if sample.empty:
        return {"events": 0, "event_mean": np.nan, "daily_clusters": 0, "daily_mean": np.nan,
                "daily_median": np.nan, "daily_win_rate": np.nan, "cluster_t_stat": np.nan,
                "p_value": np.nan, "positive_year_fraction": np.nan}
    daily = sample.groupby("date", as_index=False)["net"].mean()
    values = daily["net"].astype(float)
    n = len(values)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else np.nan
    t_stat = mean / (std / math.sqrt(n)) if n > 1 and std > 0 else np.nan
    p_value = math.erfc(abs(t_stat) / math.sqrt(2.0)) if np.isfinite(t_stat) else np.nan
    yearly = daily.assign(year=daily["date"].dt.year).groupby("year")["net"].mean()
    return {
        "events": int(len(sample)),
        "event_mean": float(sample["net"].mean()),
        "daily_clusters": int(n),
        "daily_mean": mean,
        "daily_median": float(values.median()),
        "daily_win_rate": float((values > 0).mean()),
        "cluster_t_stat": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
        "positive_year_fraction": float((yearly > 0).mean()) if len(yearly) else np.nan,
    }


def benjamini_hochberg(p_values: pd.Series, alpha: float) -> tuple[pd.Series, pd.Series]:
    p = pd.to_numeric(p_values, errors="coerce")
    valid = p.notna()
    q = pd.Series(np.nan, index=p.index, dtype=float)
    passed = pd.Series(False, index=p.index, dtype=bool)
    if not valid.any():
        return q, passed
    ordered = p[valid].sort_values()
    m = len(ordered)
    ranks = np.arange(1, m + 1)
    raw_q = ordered.to_numpy() * m / ranks
    adjusted = np.minimum.accumulate(raw_q[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)
    q.loc[ordered.index] = adjusted
    passed.loc[ordered.index] = adjusted <= float(alpha)
    return q, passed


def prefix_metrics(prefix: str, metrics: dict) -> dict:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/alpha_factory.yaml")
    parser.add_argument("--prices", default="data/processed/daily_prices.parquet")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    root = yaml.safe_load(Path(args.config).read_text())
    cfg = root.get("alpha_factory", {})
    validation = root.get("validation", {})
    train_end = validation.get("train_end", "2019-12-31")
    validation_start = validation.get("validation_start", "2020-01-01")
    validation_end = validation.get("validation_end", "2022-12-31")
    test_start = validation.get("test_start", "2023-01-01")
    fdr_alpha = float(validation.get("fdr_alpha", 0.05))
    min_train = int(validation.get("minimum_train_events", 100))
    min_validation = int(validation.get("minimum_validation_events", 30))
    min_test = int(validation.get("minimum_test_events", 30))
    min_positive_year_fraction = float(validation.get("minimum_positive_year_fraction", 0.60))

    features = build_features(pd.read_parquet(args.prices))
    rows = []
    for spec in candidate_specs(cfg):
        train = daily_cluster_statistics(period_sample(features, spec, cfg, None, train_end))
        valid = daily_cluster_statistics(period_sample(features, spec, cfg, validation_start, validation_end))
        test = daily_cluster_statistics(period_sample(features, spec, cfg, test_start, None))
        rows.append({**spec, **prefix_metrics("train", train), **prefix_metrics("validation", valid), **prefix_metrics("test", test)})

    result = pd.DataFrame(rows)
    result["train_q_value"], result["train_fdr_pass"] = benjamini_hochberg(result["train_p_value"], fdr_alpha)
    result["train_discovery_pass"] = (
        result["train_fdr_pass"]
        & result["train_events"].ge(min_train)
        & result["train_daily_mean"].gt(0)
        & result["train_positive_year_fraction"].ge(min_positive_year_fraction)
    )
    result["validation_pass"] = (
        result["train_discovery_pass"]
        & result["validation_events"].ge(min_validation)
        & result["validation_daily_mean"].gt(0)
        & result["validation_cluster_t_stat"].ge(1.645)
        & result["validation_positive_year_fraction"].ge(min_positive_year_fraction)
    )
    result["locked_test_pass"] = (
        result["validation_pass"]
        & result["test_events"].ge(min_test)
        & result["test_daily_mean"].gt(0)
        & result["test_cluster_t_stat"].ge(1.645)
        & result["test_positive_year_fraction"].ge(min_positive_year_fraction)
    )
    result["production_approved"] = False
    result = result.sort_values(
        ["locked_test_pass", "validation_pass", "train_discovery_pass", "test_daily_mean"],
        ascending=[False, False, False, False],
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "alpha_factory_oos_validation.csv", index=False)
    result[result["locked_test_pass"]].to_csv(output / "alpha_factory_locked_test_survivors.csv", index=False)

    display = ["family", "direction", "horizon", "train_events", "train_daily_mean", "train_q_value",
               "validation_events", "validation_daily_mean", "validation_cluster_t_stat",
               "test_events", "test_daily_mean", "test_cluster_t_stat", "locked_test_pass"]
    print(result[display].head(50).to_string(index=False))
    print(f"\nCandidates evaluated: {len(result)}")
    print(f"Train discoveries after FDR: {int(result['train_discovery_pass'].sum())}")
    print(f"Validation survivors: {int(result['validation_pass'].sum())}")
    print(f"Locked-test survivors: {int(result['locked_test_pass'].sum())}")
    print("Production-approved candidates: 0")


if __name__ == "__main__":
    main()
