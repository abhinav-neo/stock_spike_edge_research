\
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def performance_stats(returns: pd.Series, cost: float) -> dict:
    r = returns.dropna() - cost
    if r.empty:
        return {}
    downside = r[r < 0]
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


def candidate_masks(df: pd.DataFrame) -> list[tuple[str, pd.Series]]:
    masks = []
    return_bands = [(0.40, 0.60), (0.60, 1.00), (1.00, 10.00)]
    close_locations = [0.50, 0.75, 0.90]
    relative_volumes = [2, 5, 10]
    prices = [1, 3, 5, 10]

    for (lo, hi), cl, rv, px in product(
        return_bands, close_locations, relative_volumes, prices
    ):
        mask = (
            df["event_return"].between(lo, hi, inclusive="left")
            & (df["close_location"] >= cl)
            & (df["relative_dollar_volume"] >= rv)
            & (df["event_close"] >= px)
        )
        name = f"continuation_ret_{lo:.2f}_{hi:.2f}_cl{cl:.2f}_rv{rv}_px{px}"
        masks.append((name, mask))

    # Failed-spike masks are evaluated as short returns.
    for cl, rv, px in product([0.25, 0.40, 0.50], relative_volumes, prices):
        mask = (
            (df["close_location"] <= cl)
            & (df["relative_dollar_volume"] >= rv)
            & (df["event_close"] >= px)
        )
        name = f"failed_spike_cl{cl:.2f}_rv{rv}_px{px}"
        masks.append((name, mask))
    return masks


def evaluate(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    train_end = pd.Timestamp(cfg["train_end"])
    validation_start = pd.Timestamp(cfg["validation_start"])
    validation_end = pd.Timestamp(cfg["validation_end"])
    test_start = pd.Timestamp(cfg["test_start"])
    cost = 2 * (
        cfg["transaction_cost_bps_each_way"] + cfg["slippage_bps_each_way"]
    ) / 10_000

    df["event_date"] = pd.to_datetime(df["event_date"])
    periods = {
        "train": df["event_date"] <= train_end,
        "validation": df["event_date"].between(validation_start, validation_end),
        "test": df["event_date"] >= test_start,
    }

    rows = []
    for name, mask in candidate_masks(df):
        is_short = name.startswith("failed_spike")
        for horizon in [1, 2, 3, 5, 10, 20]:
            col = f"forward_return_{horizon}d"
            if col not in df:
                continue
            row = {"rule": name, "horizon": horizon, "side": "short" if is_short else "long"}
            valid = True
            for period, period_mask in periods.items():
                returns = df.loc[mask & period_mask, col]
                if is_short:
                    returns = -returns
                stats = performance_stats(returns, cost)
                for key, value in stats.items():
                    row[f"{period}_{key}"] = value
                if period == "train" and stats.get("n", 0) < cfg["minimum_events_per_rule"]:
                    valid = False
                if period in {"validation", "test"} and stats.get("n", 0) < cfg["minimum_events_per_test_period"]:
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


def retention_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in [1, 2, 3, 5, 10, 20, 40, 60]:
        ret_col = f"forward_return_{h}d"
        above_col = f"above_event_close_{h}d"
        if ret_col not in df:
            continue
        rows.append(
            {
                "horizon_days": h,
                "events": int(df[ret_col].notna().sum()),
                "mean_forward_return": df[ret_col].mean(),
                "median_forward_return": df[ret_col].median(),
                "pct_above_event_close": df[above_col].mean(),
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

    print("\nRetention summary")
    print(retention.to_string(index=False))
    print("\nTop candidate edges")
    cols = [
        "rule", "horizon", "side", "train_n", "train_mean_return",
        "validation_n", "validation_mean_return", "test_n",
        "test_mean_return", "test_t_stat", "robust_score"
    ]
    available = [c for c in cols if c in edges]
    print(edges[available].head(20).to_string(index=False))
    print(f"\nAccepted robust edges: {len(accepted)}")


if __name__ == "__main__":
    main()
