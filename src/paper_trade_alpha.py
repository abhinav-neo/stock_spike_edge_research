from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.alpha_factory import build_features, candidate_mask
from src.alpha_portfolio import row_to_spec, candidate_key


def latest_signals(features: pd.DataFrame, selected: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    if selected.empty or features.empty:
        return pd.DataFrame()
    latest_date = pd.to_datetime(features["date"]).max()
    rows: list[dict] = []
    for rank, (_, candidate) in enumerate(selected.iterrows(), start=1):
        spec = row_to_spec(candidate)
        mask = candidate_mask(features, spec, cfg) & pd.to_datetime(features["date"]).eq(latest_date)
        for _, bar in features.loc[mask].iterrows():
            rows.append({
                "signal_date": latest_date,
                "candidate_rank": rank,
                "candidate_key": candidate_key(spec),
                "family": spec["family"],
                "direction": spec["direction"],
                "symbol": bar["symbol"],
                "horizon": int(spec["horizon"]),
                "reference_close": float(bar["close"]),
                "avg_dollar_volume_20d": float(bar["avg_dollar_volume_20d"]),
                "relative_volume": float(bar["relative_volume"]),
            })
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values(["candidate_rank", "avg_dollar_volume_20d"], ascending=[True, False])
            .drop_duplicates(["symbol", "direction"]) 
            .reset_index(drop=True))


def build_order_blotter(signals: pd.DataFrame, paper_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signals.empty:
        return pd.DataFrame(), pd.DataFrame()
    capital = float(paper_cfg.get("paper_capital", 100000.0))
    fraction = float(paper_cfg.get("position_fraction", 0.02))
    maximum_orders = int(paper_cfg.get("maximum_orders_per_day", 3))
    maximum_symbol_fraction = float(paper_cfg.get("maximum_symbol_fraction", 0.05))
    stop_loss = float(paper_cfg.get("stop_loss", 0.25))
    target_notional = min(capital * fraction, capital * maximum_symbol_fraction)

    accepted, rejected = [], []
    for sequence, (_, signal) in enumerate(signals.iterrows(), start=1):
        item = signal.to_dict()
        if sequence > maximum_orders:
            item["rejection_reason"] = "daily_order_cap"
            rejected.append(item)
            continue
        reference = float(signal["reference_close"])
        shares = int(np.floor(target_notional / reference)) if reference > 0 else 0
        if shares < 1:
            item["rejection_reason"] = "insufficient_notional"
            rejected.append(item)
            continue
        direction = str(signal["direction"])
        stop_price = reference * (1.0 - stop_loss) if direction == "long" else reference * (1.0 + stop_loss)
        accepted.append({
            **item,
            "order_id": f"PAPER-{pd.Timestamp(signal['signal_date']).strftime('%Y%m%d')}-{sequence:03d}",
            "side": "BUY" if direction == "long" else "SELL_SHORT",
            "order_type": "MOO",
            "intended_entry_date": "NEXT_SESSION",
            "shares": shares,
            "estimated_notional": shares * reference,
            "reference_stop_price": stop_price,
            "status": "REVIEW_REQUIRED",
            "live_submission_enabled": False,
        })
    return pd.DataFrame(accepted), pd.DataFrame(rejected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a review-only paper-trading blotter; never submits live orders.")
    parser.add_argument("--config", default="config/alpha_factory.yaml")
    parser.add_argument("--prices", default="data/processed/daily_prices.parquet")
    parser.add_argument("--selected", default="reports/alpha_portfolio_selected_candidates.csv")
    parser.add_argument("--output-dir", default="reports/paper")
    args = parser.parse_args()

    root = yaml.safe_load(Path(args.config).read_text())
    factory_cfg = root.get("alpha_factory", {})
    paper_cfg = root.get("paper_trading", {})
    selected_path = Path(args.selected)
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing selected candidates: {selected_path}. Run src.run_alpha_research first.")

    selected = pd.read_csv(selected_path)
    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    features = build_features(prices)
    signals = latest_signals(features, selected, factory_cfg)
    orders, rejected = build_order_blotter(signals, paper_cfg)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    signals.to_csv(output / "paper_signals.csv", index=False)
    orders.to_csv(output / "paper_order_blotter.csv", index=False)
    rejected.to_csv(output / "paper_order_rejections.csv", index=False)
    manifest = {
        "signal_date": str(features["date"].max().date()) if len(features) else None,
        "signals": int(len(signals)),
        "orders": int(len(orders)),
        "rejections": int(len(rejected)),
        "review_required": True,
        "live_submission_enabled": False,
    }
    (output / "paper_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(pd.DataFrame([manifest]).to_string(index=False))
    if len(orders):
        print("\n" + orders[["order_id", "symbol", "side", "shares", "reference_close", "reference_stop_price", "status"]].to_string(index=False))
    print("\nNo live orders were submitted.")


if __name__ == "__main__":
    main()
