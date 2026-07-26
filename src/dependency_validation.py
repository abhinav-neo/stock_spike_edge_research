from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analyze_edges import candidate_masks


def jaccard_similarity(left: pd.Series, right: pd.Series) -> float:
    union = (left | right).sum()
    return float((left & right).sum() / union) if union else 0.0


def cluster_accepted_rules(
    events: pd.DataFrame,
    accepted: pd.DataFrame,
    validation_cfg: dict,
    similarity_threshold: float = 0.80,
) -> pd.DataFrame:
    masks = {c["rule"]: c["mask"].reset_index(drop=True) for c in candidate_masks(events.reset_index(drop=True), validation_cfg)}
    ranked = accepted.sort_values(
        ["positive_fold_fraction", "oos_trimmed_mean_return", "oos_n"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    representatives: list[dict] = []
    assigned: set[str] = set()
    cluster_id = 0
    for _, row in ranked.iterrows():
        rule = row["rule"]
        if rule in assigned or rule not in masks:
            continue
        cluster_id += 1
        members = []
        base_mask = masks[rule]
        comparable = ranked[(ranked["side"] == row["side"]) & (ranked["horizon"] == row["horizon"])]
        for candidate_rule in comparable["rule"]:
            if candidate_rule in assigned or candidate_rule not in masks:
                continue
            similarity = jaccard_similarity(base_mask, masks[candidate_rule])
            if similarity >= similarity_threshold:
                assigned.add(candidate_rule)
                members.append((candidate_rule, similarity))
        representatives.append({
            "cluster_id": cluster_id,
            "representative_rule": rule,
            "side": row["side"],
            "horizon": int(row["horizon"]),
            "cluster_size": len(members),
            "minimum_member_similarity": min((x[1] for x in members), default=1.0),
            "members": "|".join(x[0] for x in members),
            "oos_n": int(row["oos_n"]),
            "oos_mean_return": float(row["oos_mean_return"]),
            "oos_trimmed_mean_return": float(row["oos_trimmed_mean_return"]),
            "oos_win_rate": float(row["oos_win_rate"]),
            "fdr_q_value": float(row["fdr_q_value"]),
        })
    return pd.DataFrame(representatives)


def monthly_block_bootstrap(
    trades: pd.DataFrame,
    samples: int = 2000,
    seed: int = 42,
) -> dict:
    clean = trades.dropna(subset=["event_date", "net_return"]).copy()
    if clean.empty:
        return {"bootstrap_mean": np.nan, "ci_low": np.nan, "ci_high": np.nan, "probability_positive": np.nan}
    clean["month"] = pd.to_datetime(clean["event_date"]).dt.to_period("M")
    blocks = [group["net_return"].to_numpy(float) for _, group in clean.groupby("month")]
    rng = np.random.default_rng(seed)
    means = np.empty(samples)
    for i in range(samples):
        chosen = rng.integers(0, len(blocks), len(blocks))
        draw = np.concatenate([blocks[j] for j in chosen])
        means[i] = draw.mean()
    return {
        "bootstrap_mean": float(means.mean()),
        "ci_low": float(np.quantile(means, 0.025)),
        "ci_high": float(np.quantile(means, 0.975)),
        "probability_positive": float((means > 0).mean()),
    }


def rule_trades(
    events: pd.DataFrame,
    rule: str,
    horizon: int,
    side: str,
    validation_cfg: dict,
    short_borrow_bps_annual: float,
) -> pd.DataFrame:
    candidate = next(c for c in candidate_masks(events, validation_cfg) if c["rule"] == rule)
    col = f"forward_return_{horizon}d"
    selected = events.loc[candidate["mask"], ["event_date", col] + (["symbol"] if "symbol" in events else [])].copy()
    gross = selected[col] if side == "long" else -selected[col]
    round_trip_cost = 2 * (
        validation_cfg["transaction_cost_bps_each_way"] + validation_cfg["slippage_bps_each_way"]
    ) / 10_000
    borrow = short_borrow_bps_annual / 10_000 * horizon / 252 if side == "short" else 0.0
    selected["net_return"] = gross - round_trip_cost - borrow
    selected["event_date"] = pd.to_datetime(selected["event_date"])
    selected["exit_date"] = selected["event_date"] + pd.offsets.BDay(horizon)
    return selected.sort_values("event_date")


def simulate_portfolio(trades: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, dict]:
    initial_capital = float(cfg.get("initial_capital", 100000.0))
    position_fraction = float(cfg.get("position_fraction", 0.05))
    max_positions = int(cfg.get("max_concurrent_positions", 10))
    max_daily_entries = int(cfg.get("max_daily_entries", 3))
    stop_loss = cfg.get("stop_loss")
    capital = initial_capital
    active: list[dict] = []
    ledger: list[dict] = []
    daily_entries: dict[pd.Timestamp, int] = {}

    for _, trade in trades.sort_values("event_date").iterrows():
        date = pd.Timestamp(trade["event_date"])
        remaining = []
        for position in active:
            if position["exit_date"] <= date:
                capital += position["stake"] * (1.0 + position["return"])
                ledger.append(position)
            else:
                remaining.append(position)
        active = remaining
        if len(active) >= max_positions or daily_entries.get(date, 0) >= max_daily_entries:
            continue
        stake = min(initial_capital * position_fraction, capital)
        if stake <= 0:
            continue
        realized_return = float(trade["net_return"])
        if stop_loss is not None:
            realized_return = max(realized_return, -float(stop_loss))
        capital -= stake
        active.append({
            "entry_date": date,
            "exit_date": pd.Timestamp(trade["exit_date"]),
            "stake": stake,
            "return": realized_return,
            "pnl": stake * realized_return,
        })
        daily_entries[date] = daily_entries.get(date, 0) + 1

    for position in sorted(active, key=lambda x: x["exit_date"]):
        capital += position["stake"] * (1.0 + position["return"])
        ledger.append(position)
    ledger_df = pd.DataFrame(ledger)
    if ledger_df.empty:
        return ledger_df, {"initial_capital": initial_capital, "ending_capital": initial_capital, "total_return": 0.0, "trades": 0}
    ledger_df = ledger_df.sort_values("exit_date")
    ledger_df["cumulative_pnl"] = ledger_df["pnl"].cumsum()
    ledger_df["equity"] = initial_capital + ledger_df["cumulative_pnl"]
    ledger_df["drawdown"] = ledger_df["equity"] / ledger_df["equity"].cummax() - 1.0
    return ledger_df, {
        "initial_capital": initial_capital,
        "ending_capital": float(capital),
        "total_return": float(capital / initial_capital - 1.0),
        "trades": int(len(ledger_df)),
        "win_rate": float((ledger_df["return"] > 0).mean()),
        "mean_trade_return": float(ledger_df["return"].mean()),
        "worst_trade_return": float(ledger_df["return"].min()),
        "max_drawdown": float(ledger_df["drawdown"].min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--events", default="data/processed/events.parquet")
    parser.add_argument("--walk-forward", default="reports/walk_forward_summary.csv")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    events = pd.read_parquet(args.events)
    summary = pd.read_csv(args.walk_forward)
    accepted = summary[summary["accepted"].fillna(False)]
    dependency_cfg = config.get("dependency_validation", {})
    clusters = cluster_accepted_rules(
        events,
        accepted,
        config["validation"],
        float(dependency_cfg.get("jaccard_similarity_threshold", 0.80)),
    )
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    clusters.to_csv(out / "candidate_clusters.csv", index=False)

    result_rows = []
    for _, representative in clusters.iterrows():
        trades = rule_trades(
            events,
            representative["representative_rule"],
            int(representative["horizon"]),
            representative["side"],
            config["validation"],
            float(config["walk_forward"].get("short_borrow_bps_annual", 0.0)),
        )
        trades = trades[trades["event_date"] >= pd.Timestamp(config["walk_forward"]["oos_start"])]
        bootstrap = monthly_block_bootstrap(
            trades,
            int(dependency_cfg.get("bootstrap_samples", 2000)),
            int(dependency_cfg.get("random_seed", 42)),
        )
        ledger, portfolio = simulate_portfolio(trades, dependency_cfg.get("portfolio", {}))
        ledger.to_csv(out / f"portfolio_cluster_{int(representative['cluster_id'])}.csv", index=False)
        result_rows.append({**representative.to_dict(), **bootstrap, **portfolio, "research_status": "research_candidate_dependency_tested"})
    results = pd.DataFrame(result_rows)
    results.to_csv(out / "dependency_validation_summary.csv", index=False)
    print("\nDependency-aware representative candidates")
    columns = ["cluster_id", "representative_rule", "cluster_size", "horizon", "bootstrap_mean", "ci_low", "ci_high", "probability_positive", "total_return", "max_drawdown", "trades", "research_status"]
    print(results[[c for c in columns if c in results]].to_string(index=False))
    print(f"\nAccepted rows reduced from {len(accepted)} to {len(clusters)} representative clusters")
    print("Production-approved candidates: 0")


if __name__ == "__main__":
    main()
