from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.mark_to_market import apply_locate_model, trade_paths
from src.v5_mtm_research import (
    apply_round_trip_cost,
    attach_event_dates,
    build_period_trades,
    daily_equity_curve,
    load_candidate_prices,
    normalize_spy,
    performance,
    trade_statistics,
)


MARGIN_PROFILES = {
    "unconstrained": (None, None),
    "reg_t_50_30": (0.50, 0.30),
    "house_100_50": (1.00, 0.50),
    "hard_to_borrow_200_100": (2.00, 1.00),
}


def evaluate_profiles(
    candidates: pd.DataFrame,
    prices: pd.DataFrame,
    spy: pd.DataFrame,
    initial_capital: float = 100_000.0,
    stake: float = 10_000.0,
) -> pd.DataFrame:
    located, rejected = apply_locate_model(candidates, 0.70, 42)
    rows = []
    for name, (initial_margin, maintenance_margin) in MARGIN_PROFILES.items():
        paths, completed = trade_paths(
            located,
            prices,
            stop_loss=None,
            short_borrow_bps_annual=1000.0,
            initial_margin_requirement=initial_margin,
            maintenance_margin_requirement=maintenance_margin,
        )
        paths, completed = apply_round_trip_cost(paths, completed, 100.0)
        equity = daily_equity_curve(paths, completed, pd.DatetimeIndex(spy["date"]), initial_capital, stake)
        summary, _ = performance(equity, spy, initial_capital)
        rows.append(
            {
                "profile": name,
                "initial_margin_requirement": initial_margin,
                "maintenance_margin_requirement": maintenance_margin,
                "candidate_trades": len(candidates),
                "located_trades": len(located),
                "locate_rejections": len(rejected),
                "margin_liquidations": int((completed["exit_reason"] == "margin_liquidation").sum()),
                **summary,
                **trade_statistics(completed),
            }
        )
    return pd.DataFrame(rows)


def assessment(results: pd.DataFrame) -> str:
    table = "\n".join(
        [
            "| Profile | Liquidations | CAGR | Total return | Max drawdown | Sharpe | Worst trade |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        + [
            f"| {row.profile} | {int(row.margin_liquidations)} | {row.cagr:.2%} | "
            f"{row.total_return:.2%} | {row.max_drawdown:.2%} | {row.sharpe_zero_rate:.2f} | "
            f"{row.worst_trade:.1%} |"
            for row in results.itertuples()
        ]
    )
    feasible = results.loc[
        results["profile"].ne("unconstrained")
        & results["cagr"].ge(0.40)
        & results["max_drawdown"].ge(-0.25)
        & results["worst_trade"].ge(-1.0)
        & results["total_return"].gt(results["spy_total_return_aligned"])
    ]
    verdict = "PASS" if len(feasible) else "REJECT"
    return f"""# Broker Margin and Forced-Liquidation Assessment

## Verdict

**{verdict}.** No margin profile is promoted unless it reaches the locked 40% CAGR
target, keeps drawdown within 25%, bounds every loss to at most the allocated notional,
and beats aligned SPY. Margin rules are specified independently of test returns.

## Results

{table}

The profiles model position-level collateral: Reg T-like 50% initial/30% maintenance,
a 100%/50% house rule, and a 200%/100% hard-to-borrow rule. Gap-through liquidations
fill at the next available open. These scenarios still cannot reproduce discretionary
broker recalls, symbol-specific house-margin changes, or account-wide cross-margining.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress V5 shorts under broker-style forced liquidations")
    parser.add_argument("--predictions", default="reports/v5_validation_suite/model/predictions.csv")
    parser.add_argument("--features", default="data/processed/events_features_v5.parquet")
    parser.add_argument("--raw-prices-dir", default="data/raw/prices")
    parser.add_argument("--spy", default="data/raw/spy_benchmark.parquet")
    parser.add_argument("--output-dir", default="reports/margin_liquidation")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    features = pd.read_parquet(args.features)
    candidates = attach_event_dates(build_period_trades(predictions, features, "test"), features, 5)
    prices = load_candidate_prices(candidates["symbol"], Path(args.raw_prices_dir))
    spy = normalize_spy(Path(args.spy))
    results = evaluate_profiles(candidates, prices, spy)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "margin_profiles.csv", index=False)
    (output / "margin_profiles.json").write_text(
        json.dumps(results.to_dict("records"), indent=2, default=str), encoding="utf-8"
    )
    (output / "ASSESSMENT.md").write_text(assessment(results), encoding="utf-8")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
