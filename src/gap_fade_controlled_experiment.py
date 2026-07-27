from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.alpha_factory import build_features
from src.alpha_portfolio import (
    apply_capacity,
    build_trades,
    execute_trade,
    portfolio_summary,
    select_representatives,
)


def select_gap_fade_short(survivors: pd.DataFrame, maximum_candidates: int = 2) -> pd.DataFrame:
    """Keep only the strongest distinct gap-fade short candidates."""
    eligible = survivors[
        survivors["family"].eq("gap_fade")
        & survivors["direction"].eq("short")
    ].copy()
    return select_representatives(eligible, maximum_candidates, maximum_candidates)


def attach_signal_quality(trades: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    quality_columns = [
        "symbol",
        "date",
        "close",
        "previous_close",
        "avg_dollar_volume_20d",
        "relative_volume",
    ]
    quality = features[quality_columns].rename(
        columns={
            "date": "signal_date",
            "close": "signal_close",
            "previous_close": "signal_previous_close",
        }
    )
    merged = trades.merge(quality, on=["symbol", "signal_date"], how="left")
    merged["signal_close_ratio"] = (
        merged["signal_close"] / merged["signal_previous_close"]
    )
    return merged


def apply_trade_quality_filters(
    trades: pd.DataFrame,
    minimum_entry_price: float = 5.0,
    maximum_entry_price: float = 1000.0,
    minimum_avg_dollar_volume: float = 10_000_000.0,
    minimum_close_ratio: float = 0.20,
    maximum_close_ratio: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reject low-liquidity and likely corporate-action/data-discontinuity trades."""
    if trades.empty:
        return trades.copy(), trades.copy()

    accepted: list[dict] = []
    rejected: list[dict] = []
    for _, trade in trades.iterrows():
        reasons: list[str] = []
        entry_price = float(trade["entry_price"])
        adv = pd.to_numeric(pd.Series([trade.get("avg_dollar_volume_20d")]), errors="coerce").iloc[0]
        ratio = pd.to_numeric(pd.Series([trade.get("signal_close_ratio")]), errors="coerce").iloc[0]

        if entry_price < minimum_entry_price:
            reasons.append("entry_price_below_minimum")
        if entry_price > maximum_entry_price:
            reasons.append("entry_price_above_maximum")
        if not np.isfinite(adv) or float(adv) < minimum_avg_dollar_volume:
            reasons.append("insufficient_liquidity")
        if not np.isfinite(ratio) or not (minimum_close_ratio <= float(ratio) <= maximum_close_ratio):
            reasons.append("possible_corporate_action_or_bad_price")

        item = trade.to_dict()
        if reasons:
            item["rejection_reason"] = ";".join(reasons)
            rejected.append(item)
        else:
            accepted.append(item)

    return pd.DataFrame(accepted), pd.DataFrame(rejected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a controlled gap-fade short portfolio experiment.")
    parser.add_argument("--config", default="config/alpha_factory.yaml")
    parser.add_argument("--prices", default="data/processed/daily_prices.parquet")
    parser.add_argument("--survivors", default="reports/alpha_factory_locked_test_survivors.csv")
    parser.add_argument("--output-dir", default="reports/gap_fade_controlled")
    args = parser.parse_args()

    root = yaml.safe_load(Path(args.config).read_text())
    factory_cfg = root.get("alpha_factory", {})
    validation_cfg = root.get("validation", {})
    experiment_cfg = root.get("gap_fade_controlled", {})

    survivors = pd.read_csv(args.survivors)
    selected = select_gap_fade_short(
        survivors,
        int(experiment_cfg.get("maximum_candidates", 2)),
    )
    if selected.empty:
        raise ValueError("No locked-test gap_fade short survivors were available.")

    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    features = build_features(prices)

    candidate_trades = build_trades(
        features,
        selected,
        factory_cfg,
        validation_cfg.get("test_start", "2023-01-01"),
    )
    enriched = attach_signal_quality(candidate_trades, features)
    quality_accepted, quality_rejected = apply_trade_quality_filters(
        enriched,
        minimum_entry_price=float(experiment_cfg.get("minimum_entry_price", 5.0)),
        maximum_entry_price=float(experiment_cfg.get("maximum_entry_price", 1000.0)),
        minimum_avg_dollar_volume=float(experiment_cfg.get("minimum_avg_dollar_volume", 10_000_000.0)),
        minimum_close_ratio=float(experiment_cfg.get("minimum_close_ratio", 0.20)),
        maximum_close_ratio=float(experiment_cfg.get("maximum_close_ratio", 5.0)),
    )
    accepted, capacity_rejected = apply_capacity(
        quality_accepted,
        int(experiment_cfg.get("max_daily_entries", 2)),
        int(experiment_cfg.get("max_concurrent_positions", 6)),
    )

    completed = pd.DataFrame([
        execute_trade(
            row,
            prices,
            float(experiment_cfg.get("stop_loss", 0.20)),
            float(experiment_cfg.get("round_trip_cost_bps", 100.0)),
        )
        for _, row in accepted.iterrows()
    ])
    curve, summary = portfolio_summary(
        completed,
        float(experiment_cfg.get("initial_capital", 100_000.0)),
        float(experiment_cfg.get("position_fraction", 0.02)),
    )
    summary.update({
        "selected_candidates": int(len(selected)),
        "candidate_trades": int(len(candidate_trades)),
        "quality_rejections": int(len(quality_rejected)),
        "capacity_rejections": int(len(capacity_rejected)),
        "experiment": "gap_fade_short_controlled",
        "production_approved": False,
    })

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output / "selected_candidates.csv", index=False)
    completed.to_csv(output / "trades.csv", index=False)
    curve.to_csv(output / "equity.csv", index=False)
    quality_rejected.to_csv(output / "quality_rejections.csv", index=False)
    capacity_rejected.to_csv(output / "capacity_rejections.csv", index=False)
    pd.DataFrame([summary]).to_csv(output / "summary.csv", index=False)

    print(pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_string())
    print("\nControlled experiment only. Live submission remains disabled.")


if __name__ == "__main__":
    main()
