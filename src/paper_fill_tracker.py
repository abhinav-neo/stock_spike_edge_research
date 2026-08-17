from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _next_bar(prices: pd.DataFrame, symbol: str, signal_date: pd.Timestamp) -> pd.Series | None:
    rows = prices[(prices["symbol"] == symbol) & (prices["date"] > signal_date)].sort_values("date")
    if rows.empty:
        return None
    return rows.iloc[0]


def process_orders(orders: pd.DataFrame, prices: pd.DataFrame, paper_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    if orders.empty:
        return pd.DataFrame(), pd.DataFrame()

    stop_loss = float(paper_cfg.get("stop_loss", 0.25))
    slippage_bps = float(paper_cfg.get("paper_fill_slippage_bps", 0.0))
    fills: list[dict] = []
    pending: list[dict] = []

    for _, order in orders.iterrows():
        item = order.to_dict()
        signal_date = pd.Timestamp(order["signal_date"])
        bar = _next_bar(prices, str(order["symbol"]), signal_date)
        if bar is None:
            item.update({"fill_status": "PENDING_NEXT_SESSION", "live_submission_enabled": False})
            pending.append(item)
            continue

        raw_open = float(bar["open"])
        side = str(order["side"])
        adverse = slippage_bps / 10000.0
        fill_price = raw_open * (1.0 + adverse) if side == "BUY" else raw_open * (1.0 - adverse)
        reference = float(order["reference_close"])
        signed_slippage = fill_price / reference - 1.0
        if side == "SELL_SHORT":
            signed_slippage = -signed_slippage
        direction = str(order["direction"])
        stop_price = fill_price * (1.0 - stop_loss) if direction == "long" else fill_price * (1.0 + stop_loss)
        shares = int(order["shares"])
        fills.append({
            **item,
            "fill_status": "PAPER_FILLED",
            "fill_date": pd.Timestamp(bar["date"]),
            "market_open": raw_open,
            "fill_price": fill_price,
            "fill_notional": shares * fill_price,
            "entry_slippage_return": signed_slippage,
            "entry_slippage_bps": signed_slippage * 10000.0,
            "active_stop_price": stop_price,
            "position_status": "OPEN",
            "live_submission_enabled": False,
        })

    return pd.DataFrame(fills), pd.DataFrame(pending)


def mark_positions(fills: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    if fills.empty:
        return pd.DataFrame()
    rows: list[dict] = []
    for _, fill in fills.iterrows():
        symbol_rows = prices[(prices["symbol"] == fill["symbol"]) & (prices["date"] >= pd.Timestamp(fill["fill_date"]))].sort_values("date")
        latest = symbol_rows.iloc[-1]
        mark = float(latest["close"])
        entry = float(fill["fill_price"])
        direction = str(fill["direction"])
        gross_return = mark / entry - 1.0
        if direction == "short":
            gross_return = -gross_return
        unrealized_pnl = gross_return * float(fill["fill_notional"])
        rows.append({
            "order_id": fill["order_id"],
            "symbol": fill["symbol"],
            "direction": direction,
            "shares": int(fill["shares"]),
            "fill_date": fill["fill_date"],
            "fill_price": entry,
            "mark_date": pd.Timestamp(latest["date"]),
            "mark_price": mark,
            "active_stop_price": float(fill["active_stop_price"]),
            "gross_return": gross_return,
            "unrealized_pnl": unrealized_pnl,
            "position_status": "OPEN",
            "live_submission_enabled": False,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper fills from actual next-session opens; never submits orders.")
    parser.add_argument("--config", default="config/alpha_factory.yaml")
    parser.add_argument("--prices", default="data/processed/daily_prices.parquet")
    parser.add_argument("--orders", default="reports/paper/paper_order_blotter.csv")
    parser.add_argument("--output-dir", default="reports/paper")
    args = parser.parse_args()

    orders_path = Path(args.orders)
    if not orders_path.exists():
        raise FileNotFoundError(f"Missing paper blotter: {orders_path}. Run src.paper_trade_alpha first.")

    root = yaml.safe_load(Path(args.config).read_text())
    paper_cfg = root.get("paper_trading", {})
    orders = pd.read_csv(orders_path)
    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    if "signal_date" in orders:
        orders["signal_date"] = pd.to_datetime(orders["signal_date"])

    fills, pending = process_orders(orders, prices, paper_cfg)
    positions = mark_positions(fills, prices)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fills.to_csv(output / "paper_fills.csv", index=False)
    pending.to_csv(output / "paper_pending_orders.csv", index=False)
    positions.to_csv(output / "paper_open_positions.csv", index=False)

    summary = {
        "orders": int(len(orders)),
        "fills": int(len(fills)),
        "pending": int(len(pending)),
        "open_positions": int(len(positions)),
        "total_unrealized_pnl": float(positions["unrealized_pnl"].sum()) if len(positions) else 0.0,
        "live_submission_enabled": False,
    }
    pd.DataFrame([summary]).to_csv(output / "paper_fill_summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))
    if len(fills):
        print("\n" + fills[["order_id", "symbol", "side", "shares", "fill_date", "fill_price", "entry_slippage_bps", "fill_status"]].to_string(index=False))
    if len(pending):
        print("\nPending orders:\n" + pending[["order_id", "symbol", "signal_date", "fill_status"]].to_string(index=False))
    print("\nNo live orders were submitted.")


if __name__ == "__main__":
    main()
