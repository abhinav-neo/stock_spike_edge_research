from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def evaluate_position(fill: pd.Series, prices: pd.DataFrame, exit_slippage_bps: float) -> dict:
    bars = prices[(prices["symbol"] == fill["symbol"]) & (prices["date"] >= pd.Timestamp(fill["fill_date"]))].sort_values("date")
    if bars.empty:
        return {**fill.to_dict(), "position_status": "OPEN"}

    direction = str(fill["direction"])
    entry = float(fill["fill_price"])
    stop = float(fill["active_stop_price"])
    horizon = int(fill["horizon"])
    exit_bar = None
    exit_price = None
    exit_reason = None
    fill_type = "none"

    for i, (_, bar) in enumerate(bars.iterrows()):
        if direction == "long":
            if float(bar["open"]) <= stop:
                exit_bar, exit_price, exit_reason, fill_type = bar, float(bar["open"]), "STOP", "gap_open"
                break
            if float(bar["low"]) <= stop:
                exit_bar, exit_price, exit_reason, fill_type = bar, stop, "STOP", "intraday"
                break
        else:
            if float(bar["open"]) >= stop:
                exit_bar, exit_price, exit_reason, fill_type = bar, float(bar["open"]), "STOP", "gap_open"
                break
            if float(bar["high"]) >= stop:
                exit_bar, exit_price, exit_reason, fill_type = bar, stop, "STOP", "intraday"
                break
        if i >= horizon - 1:
            exit_bar, exit_price, exit_reason = bar, float(bar["close"]), "TIME"
            break

    if exit_bar is None:
        latest = bars.iloc[-1]
        mark = float(latest["close"])
        gross = mark / entry - 1.0
        if direction == "short":
            gross = -gross
        return {**fill.to_dict(), "mark_date": latest["date"], "mark_price": mark,
                "unrealized_return": gross, "unrealized_pnl": gross * float(fill["fill_notional"]),
                "position_status": "OPEN", "live_submission_enabled": False}

    adverse = float(exit_slippage_bps) / 10000.0
    adjusted = exit_price * (1.0 - adverse) if direction == "long" else exit_price * (1.0 + adverse)
    gross = adjusted / entry - 1.0
    if direction == "short":
        gross = -gross
    pnl = gross * float(fill["fill_notional"])
    return {**fill.to_dict(), "exit_date": pd.Timestamp(exit_bar["date"]), "raw_exit_price": exit_price,
            "exit_price": adjusted, "exit_reason": exit_reason, "stop_fill_type": fill_type,
            "realized_return": gross, "realized_pnl": pnl, "position_status": "CLOSED",
            "live_submission_enabled": False}


def process_positions(fills: pd.DataFrame, prices: pd.DataFrame, paper_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fills.empty:
        return pd.DataFrame(), pd.DataFrame()
    results = [evaluate_position(row, prices, float(paper_cfg.get("paper_exit_slippage_bps", 5.0))) for _, row in fills.iterrows()]
    frame = pd.DataFrame(results)
    return frame[frame["position_status"].eq("OPEN")].copy(), frame[frame["position_status"].eq("CLOSED")].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Track paper-position exits and realized PnL; never submits orders.")
    parser.add_argument("--config", default="config/alpha_factory.yaml")
    parser.add_argument("--prices", default="data/processed/daily_prices.parquet")
    parser.add_argument("--fills", default="reports/paper/paper_fills.csv")
    parser.add_argument("--output-dir", default="reports/paper")
    args = parser.parse_args()

    fills_path = Path(args.fills)
    if not fills_path.exists():
        raise FileNotFoundError(f"Missing fills: {fills_path}. Run src.paper_fill_tracker first.")
    root = yaml.safe_load(Path(args.config).read_text())
    fills = pd.read_csv(fills_path)
    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    if len(fills):
        fills["fill_date"] = pd.to_datetime(fills["fill_date"])
    open_positions, closed = process_positions(fills, prices, root.get("paper_trading", {}))

    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    open_positions.to_csv(output / "paper_open_positions.csv", index=False)
    closed.to_csv(output / "paper_closed_positions.csv", index=False)
    summary = {"fills": int(len(fills)), "open_positions": int(len(open_positions)), "closed_positions": int(len(closed)),
               "realized_pnl": float(closed["realized_pnl"].sum()) if len(closed) else 0.0,
               "unrealized_pnl": float(open_positions["unrealized_pnl"].sum()) if len(open_positions) else 0.0,
               "live_submission_enabled": False}
    pd.DataFrame([summary]).to_csv(output / "paper_lifecycle_summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))
    print("\nNo live orders were submitted.")


if __name__ == "__main__":
    main()
