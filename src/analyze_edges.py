from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


DEFAULT_GRID = {
    "continuation_return_bands": [[0.40, 0.60], [0.60, 1.00], [1.00, 10.00]],
    "continuation_close_locations": [0.50, 0.75, 0.90],
    "failed_spike_close_locations": [0.25, 0.40, 0.50],
    "relative_volumes": [2, 5, 10],
    "minimum_prices": [1, 3, 5, 10],
}


def performance_stats(returns: pd.Series, cost: float) -> dict:
    r = returns.dropna() - cost
    if r.empty:
        return {
            "n": 0,
            "mean_return": np.nan,
            "median_return": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "t_stat": np.nan,
            "worst_trade": np.nan,
            "best_trade": np.nan,
        }
    return {
        "n": int(len(r)),
        "mean_return": float(r.mean()),
        "median_return": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": float(r[r > 0].sum() / abs(r[r < 0].sum()))
        if (r < 0).any()
        else np.inf,
        "t_stat": float(r.mean() / (r.std(ddof=1) / np.sqrt(len(r))))
        if len(r) > 1 and r.std(ddof=1) > 0
        else np.nan,
        "worst_trade": float(r.min()),
        "best_trade": float(r.max()),
    }


def load_parameter_grid(cfg: dict) -> dict:
    grid = {**DEFAULT_GRID, **cfg.get("parameter_grid", {})}
    for key, values in grid.items():
        if not values:
            raise ValueError(f"parameter_grid.{key} must contain at least one value")
    return grid


def candidate_masks(df: pd.DataFrame, cfg: dict | None = None) -> list[dict]:
    grid = load_parameter_grid(cfg or {})
    candidates: list[dict] = []

    for band, cl, rv, px in product(
        grid["continuation_return_bands"],
        grid["continuation_close_locations"],
        grid["relative_volumes"],
        grid["minimum_prices"],
    ):
        lo, hi = map(float, band)
        cl = float(cl)
        rv = float(rv)
        px = float(px)
        mask = (
            df["event_return"].between(lo, hi, inclusive="left")
            & (df["close_location"] >= cl)
            & (df["relative_dollar_volume"] >= rv)
            & (df["event_close"] >= px)
        )
        candidates.append(
            {
                "rule": f"continuation_ret_{lo:.2f}_{hi:.2f}_cl{cl:.2f}_rv{rv:g}_px{px:g}",
                "mask": mask,
                "side": "long",
                "return_low": lo,
                "return_high": hi,
                "close_location": cl,
                "relative_volume": rv,
                "minimum_price": px,
            }
        )

    for cl, rv, px in product(
        grid["failed_spike_close_locations"],
        grid["relative_volumes"],
        grid["minimum_prices"],
    ):
        cl = float(cl)
        rv = float(rv)
        px = float(px)
        mask = (
            (df["close_location"] <= cl)
            & (df["relative_dollar_volume"] >= rv)
            & (df["event_close"] >= px)
        )
        candidates.append(
            {
                "rule": f"failed_spike_cl{cl:.2f}_rv{rv:g}_px{px:g}",
                "mask": mask,
                "side": "short",
                "return_low": np.nan,
                "return_high": np.nan,
                "close_location": cl,
                "relative_volume": rv,
                "minimum_price": px,
            }
        )
    return candidates


def evaluate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    train_end = pd.Timestamp(cfg["train_end"])
    validation_start = pd.Timestamp(cfg["validation_start"])
    validation_end = pd.Timestamp(cfg["validation_end"])
    test_start = pd.Timestamp(cfg["test_start"])
    cost = 2 * (
        cfg["transaction_cost_bps_each_way"] + cfg["slippage_bps_each_way"]
    ) / 10_000

    df = df.copy()
    df["event_date"] = pd.to_datetime(df["event_date"])
    periods = {
        "train": df["event_date"] <= train_end,
        "validation": df["event_date"].between(validation_start, validation_end),
        "test": df["event_date"] >= test_start,
    }

    rows = []
    horizons = cfg.get("horizons", [1, 2, 3, 5, 10, 20, 40, 60])
    for candidate in candidate_masks(df, cfg):
        for horizon in horizons:
            col = f"forward_return_{horizon}d"
            if col not in df:
                continue
            row = {key: value for key, value in candidate.items() if key != "mask"}
            row["horizon"] = horizon
            valid = True
            for period, period_mask in periods.items():
                returns = df.loc[candidate["mask"] & period_mask, col]
                if candidate["side"] == "short":
                    returns = -returns
                stats = performance_stats(returns, cost)
                for key, value in stats.items():
                    row[f"{period}_{key}"] = value
                minimum = (
                    cfg["minimum_events_per_rule"]
                    if period == "train"
                    else cfg["minimum_events_per_test_period"]
                )
                if stats["n"] < minimum:
                    valid = False
            row["sample_size_pass"] = valid
            row["stable_positive"] = bool(
                valid
                and row.get("train_mean_return", -1) > 0
                and row.get("validation_mean_return", -1) > 0
                and row.get("test_mean_return", -1) > 0
            )
            rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        result["robust_score"] = (
            result[["train_mean_return", "validation_mean_return", "test_mean_return"]]
            .min(axis=1)
            * np.log1p(result[["train_n", "validation_n", "test_n"]].min(axis=1))
        )
        result = result.sort_values(
            ["stable_positive", "robust_score"], ascending=[False, False]
        )
    return result


def parameter_stability(edges: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for parameter in ["close_location", "relative_volume", "minimum_price", "horizon"]:
        grouped = (
            edges.groupby(["side", parameter], dropna=False)
            .agg(
                rules=("rule", "count"),
                passing_rules=("sample_size_pass", "sum"),
                median_robust_score=("robust_score", "median"),
                median_test_mean_return=("test_mean_return", "median"),
                median_test_win_rate=("test_win_rate", "median"),
                median_test_t_stat=("test_t_stat", "median"),
            )
            .reset_index()
            .rename(columns={parameter: "parameter_value"})
        )
        grouped.insert(1, "parameter", parameter)
        frames.append(grouped)
    return pd.concat(frames, ignore_index=True)


def write_heatmap_tables(edges: pd.DataFrame, output_dir: Path) -> None:
    failed = edges[edges["side"] == "short"]
    for horizon in sorted(failed["horizon"].dropna().unique()):
        matrix = failed[failed["horizon"] == horizon].pivot_table(
            index="close_location",
            columns="relative_volume",
            values="robust_score",
            aggfunc="median",
        )
        matrix.to_csv(output_dir / f"heatmap_failed_spike_h{int(horizon)}_cl_vs_rv.csv")


def retention_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in [1, 2, 3, 5, 10, 20, 40, 60]:
        ret_col = f"forward_return_{h}d"
        above_col = f"above_entry_open_{h}d"
        if ret_col not in df:
            continue
        above_series = (
            df[above_col]
            if above_col in df
            else df.get(f"above_event_close_{h}d")
        )
        rows.append(
            {
                "horizon_days": h,
                "events": int(df[ret_col].notna().sum()),
                "mean_forward_return": df[ret_col].mean(),
                "median_forward_return": df[ret_col].median(),
                "pct_above_entry_open": above_series.mean()
                if above_series is not None
                else np.nan,
                "pct_down_10pct_or_more": (df[ret_col] <= -0.10).mean(),
                "pct_up_10pct_or_more": (df[ret_col] >= 0.10).mean(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--events", default="data/processed/events.parquet")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["validation"]
    df = pd.read_parquet(args.events)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    retention = retention_summary(df)
    retention.to_csv(out / "retention_summary.csv", index=False)

    edges = evaluate(df, cfg)
    edges.to_csv(out / "candidate_edges.csv", index=False)

    accepted = edges[
        edges["stable_positive"].fillna(False)
        & (edges.get("test_t_stat", pd.Series(index=edges.index, dtype=float)) >= 1.5)
    ]
    accepted.to_csv(out / "accepted_edges.csv", index=False)

    parameter_stability(edges).to_csv(out / "parameter_stability.csv", index=False)
    write_heatmap_tables(edges, out)

    print("\nRetention summary")
    print(retention.to_string(index=False))
    print("\nTop candidate edges")
    cols = [
        "rule", "horizon", "side", "sample_size_pass", "train_n",
        "train_mean_return", "train_median_return", "train_win_rate", "train_t_stat",
        "validation_n", "validation_mean_return", "validation_median_return",
        "validation_win_rate", "validation_t_stat", "test_n", "test_mean_return",
        "test_median_return", "test_win_rate", "test_t_stat", "robust_score",
    ]
    print(edges[[c for c in cols if c in edges]].head(20).to_string(index=False))
    print(f"\nAccepted robust edges: {len(accepted)}")
    print(f"Parameter combinations evaluated: {len(edges)}")


if __name__ == "__main__":
    main()
