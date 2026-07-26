from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.mark_to_market import (
    apply_locate_model,
    build_rule_trades,
    mark_to_market_portfolio,
    trade_paths,
)


def scenario_grid(cfg: dict) -> list[tuple[float, float]]:
    borrow_values = cfg.get("borrow_bps_grid", [0, 500, 1000, 2000, 5000])
    locate_values = cfg.get("locate_probability_grid", [1.0, 0.8, 0.6, 0.4])
    return [(float(b), float(p)) for b in borrow_values for p in locate_values]


def yearly_trade_summary(trades: pd.DataFrame, stake: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(
            columns=[
                "entry_year",
                "trades",
                "mean_net_return",
                "median_net_return",
                "win_rate",
                "total_pnl",
                "worst_trade",
                "stop_rate",
                "gap_stop_rate",
            ]
        )

    frame = trades.copy()
    frame["entry_year"] = pd.to_datetime(frame["entry_date"]).dt.year
    frame["pnl"] = frame["net_return"] * float(stake)
    frame["win"] = frame["net_return"] > 0
    frame["stopped"] = frame.get("exit_reason", "") == "stop"
    frame["gap_stop"] = frame.get("stop_fill_type", "") == "gap_open"

    return (
        frame.groupby("entry_year", as_index=False)
        .agg(
            trades=("net_return", "size"),
            mean_net_return=("net_return", "mean"),
            median_net_return=("net_return", "median"),
            win_rate=("win", "mean"),
            total_pnl=("pnl", "sum"),
            worst_trade=("net_return", "min"),
            stop_rate=("stopped", "mean"),
            gap_stop_rate=("gap_stop", "mean"),
        )
        .sort_values("entry_year")
    )


def temporal_stability(yearly: pd.DataFrame) -> dict:
    if yearly.empty:
        return {
            "years": 0,
            "positive_year_fraction": np.nan,
            "profitable_year_fraction": np.nan,
            "worst_year_pnl": np.nan,
            "best_year_pnl": np.nan,
        }
    return {
        "years": int(len(yearly)),
        "positive_year_fraction": float((yearly["mean_net_return"] > 0).mean()),
        "profitable_year_fraction": float((yearly["total_pnl"] > 0).mean()),
        "worst_year_pnl": float(yearly["total_pnl"].min()),
        "best_year_pnl": float(yearly["total_pnl"].max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--events", default="data/processed/events.parquet")
    parser.add_argument("--prices", default="data/processed/daily_prices.parquet")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    events = pd.read_parquet(args.events)
    prices = pd.read_parquet(args.prices)
    mtm_cfg = config.get("mark_to_market", {})
    stress_cfg = config.get("mtm_stress_test", {})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    initial_capital = float(mtm_cfg.get("initial_capital", 100000))
    position_fraction = float(mtm_cfg.get("position_fraction", 0.05))
    stake = initial_capital * position_fraction
    seed = int(mtm_cfg.get("locate_random_seed", 42))
    rows: list[dict] = []

    for spec in mtm_cfg.get("candidates", []):
        candidate_trades = build_rule_trades(
            events,
            spec["rule"],
            int(spec["horizon"]),
            config["validation"],
        )
        name = f"{spec['rule']}_{spec['horizon']}d"

        for borrow_bps, locate_probability in scenario_grid(stress_cfg):
            located, rejected = apply_locate_model(candidate_trades, locate_probability, seed)
            paths, completed = trade_paths(
                located,
                prices,
                mtm_cfg.get("stop_loss"),
                borrow_bps,
            )
            _, summary = mark_to_market_portfolio(
                paths,
                completed,
                initial_capital,
                position_fraction,
            )
            yearly = yearly_trade_summary(completed, stake)
            stability = temporal_stability(yearly)

            scenario_name = f"{name}_borrow{int(borrow_bps)}_locate{int(round(locate_probability * 100))}"
            yearly.to_csv(output / f"mtm_yearly_{scenario_name}.csv", index=False)
            rows.append(
                {
                    "rule": spec["rule"],
                    "horizon": int(spec["horizon"]),
                    "borrow_bps_annual": borrow_bps,
                    "locate_probability": locate_probability,
                    "candidate_trades": int(len(candidate_trades)),
                    "locate_rejections": int(len(rejected)),
                    **summary,
                    **stability,
                    "research_status": "scenario_and_temporal_stress_tested",
                    "production_approved": False,
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(output / "mtm_stress_test_summary.csv", index=False)

    display_columns = [
        "rule",
        "horizon",
        "borrow_bps_annual",
        "locate_probability",
        "trades",
        "total_return",
        "cagr",
        "max_drawdown",
        "sharpe",
        "active_day_sharpe",
        "positive_year_fraction",
        "profitable_year_fraction",
        "worst_year_pnl",
    ]
    available = [c for c in display_columns if c in result.columns]
    print(result[available].to_string(index=False))
    print("\nProduction-approved candidates: 0")


if __name__ == "__main__":
    main()
