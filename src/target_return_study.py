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
    target_cfg = config.get("target_return_study", {})

    initial_capital = float(mtm_cfg.get("initial_capital", 100000))
    stop_loss = mtm_cfg.get("stop_loss")
    borrow_bps = float(target_cfg.get("short_borrow_bps_annual", mtm_cfg.get("short_borrow_bps_annual", 1000)))
    locate_probability = float(target_cfg.get("locate_availability_probability", mtm_cfg.get("locate_availability_probability", 0.70)))
    seed = int(mtm_cfg.get("locate_random_seed", 42))
    target_cagr = float(target_cfg.get("target_cagr", 0.40))
    max_drawdown_limit = float(target_cfg.get("maximum_acceptable_drawdown", 0.25))
    max_exposure_limit = float(target_cfg.get("maximum_gross_exposure", 1.00))
    position_grid = [float(x) for x in target_cfg.get("position_fraction_grid", [0.02, 0.05, 0.08, 0.10, 0.15, 0.20])]

    rows: list[dict] = []
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for spec in mtm_cfg.get("candidates", []):
        candidates = build_rule_trades(events, spec["rule"], int(spec["horizon"]), config["validation"])
        located, rejected = apply_locate_model(candidates, locate_probability, seed)
        paths, completed = trade_paths(located, prices, stop_loss, borrow_bps)

        for position_fraction in position_grid:
            equity, summary = mark_to_market_portfolio(
                paths,
                completed,
                initial_capital,
                position_fraction,
            )
            cagr = float(summary.get("cagr", np.nan))
            max_drawdown = abs(float(summary.get("max_drawdown", np.nan)))
            max_exposure = float(summary.get("maximum_gross_exposure", np.nan))
            target_met = bool(np.isfinite(cagr) and cagr >= target_cagr)
            drawdown_pass = bool(np.isfinite(max_drawdown) and max_drawdown <= max_drawdown_limit)
            exposure_pass = bool(np.isfinite(max_exposure) and max_exposure <= max_exposure_limit)
            feasible = target_met and drawdown_pass and exposure_pass

            rows.append({
                "rule": spec["rule"],
                "horizon": int(spec["horizon"]),
                "position_fraction": position_fraction,
                "target_cagr": target_cagr,
                "borrow_bps_annual": borrow_bps,
                "locate_probability": locate_probability,
                "candidate_trades": int(len(candidates)),
                "locate_rejections": int(len(rejected)),
                **summary,
                "target_met": target_met,
                "drawdown_pass": drawdown_pass,
                "exposure_pass": exposure_pass,
                "target_feasible": feasible,
                "research_status": "target_return_capacity_tested",
                "production_approved": False,
            })

            name = f"{spec['rule']}_{spec['horizon']}d_pf{int(round(position_fraction * 1000)):03d}"
            equity.to_csv(output / f"target_equity_{name}.csv", index=False)

    result = pd.DataFrame(rows)
    result.to_csv(output / "target_return_study.csv", index=False)

    display_columns = [
        "rule",
        "horizon",
        "position_fraction",
        "cagr",
        "total_return",
        "max_drawdown",
        "maximum_gross_exposure",
        "sharpe",
        "target_met",
        "drawdown_pass",
        "exposure_pass",
        "target_feasible",
    ]
    available = [column for column in display_columns if column in result.columns]
    print(result[available].to_string(index=False))
    print(f"\nTarget CAGR: {target_cagr:.1%}")
    print(f"Feasible candidates: {int(result['target_feasible'].sum()) if not result.empty else 0}")
    print("Production-approved candidates: 0")


if __name__ == "__main__":
    main()
