from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.analyze_edges import candidate_masks

REQUIRED_PRICE_COLUMNS = {"symbol", "date", "open", "high", "low", "close"}


def validate_prices(prices: pd.DataFrame) -> None:
    missing = REQUIRED_PRICE_COLUMNS - set(prices.columns)
    if missing:
        raise ValueError(f"daily price data missing required columns: {sorted(missing)}")
    if prices.duplicated(["symbol", "date"]).any():
        raise ValueError("daily price data contains duplicate symbol/date rows")


def build_rule_trades(events: pd.DataFrame, rule: str, horizon: int, validation_cfg: dict) -> pd.DataFrame:
    candidates = {c["rule"]: c for c in candidate_masks(events, validation_cfg)}
    if rule not in candidates:
        raise ValueError(f"unknown rule: {rule}")
    selected = events.loc[candidates[rule]["mask"]].copy()
    required = {"symbol", "event_date"}
    missing = required - set(selected.columns)
    if missing:
        raise ValueError(f"events missing required columns: {sorted(missing)}")
    selected["event_date"] = pd.to_datetime(selected["event_date"])
    selected["horizon"] = int(horizon)
    return selected[["symbol", "event_date", "horizon"]].sort_values("event_date")


def trade_paths(trades: pd.DataFrame, prices: pd.DataFrame, stop_loss: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_prices(prices)
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["symbol", "date"])
    path_rows: list[dict] = []
    trade_rows: list[dict] = []
    for trade_id, trade in trades.reset_index(drop=True).iterrows():
        symbol = trade["symbol"]
        after = px[(px["symbol"] == symbol) & (px["date"] > trade["event_date"])].head(int(trade["horizon"]))
        if after.empty:
            continue
        entry = float(after.iloc[0]["open"])
        if not np.isfinite(entry) or entry <= 0:
            continue
        exit_reason = "time"
        exit_price = float(after.iloc[-1]["close"])
        exit_date = after.iloc[-1]["date"]
        mae = 0.0
        mfe = 0.0
        for day_number, (_, row) in enumerate(after.iterrows(), start=1):
            close_ret = (entry - float(row["close"])) / entry
            adverse = (entry - float(row["high"])) / entry
            favorable = (entry - float(row["low"])) / entry
            mae = min(mae, adverse)
            mfe = max(mfe, favorable)
            stopped = stop_loss is not None and float(row["high"]) >= entry * (1.0 + float(stop_loss))
            mark_ret = -float(stop_loss) if stopped else close_ret
            path_rows.append({"trade_id": trade_id, "symbol": symbol, "date": row["date"], "day": day_number, "mark_return": mark_ret})
            if stopped:
                exit_reason = "stop"
                exit_price = entry * (1.0 + float(stop_loss))
                exit_date = row["date"]
                break
        net_return = (entry - exit_price) / entry
        trade_rows.append({"trade_id": trade_id, "symbol": symbol, "entry_date": after.iloc[0]["date"], "exit_date": exit_date, "entry_price": entry, "exit_price": exit_price, "net_return": net_return, "mae": mae, "mfe": mfe, "exit_reason": exit_reason})
    return pd.DataFrame(path_rows), pd.DataFrame(trade_rows)


def mark_to_market_portfolio(paths: pd.DataFrame, trades: pd.DataFrame, initial_capital: float = 100000.0, position_fraction: float = 0.05) -> tuple[pd.DataFrame, dict]:
    if paths.empty or trades.empty:
        return pd.DataFrame(), {"initial_capital": initial_capital, "ending_equity": initial_capital, "total_return": 0.0, "max_drawdown": 0.0, "trades": 0}
    stake = initial_capital * position_fraction
    merged = paths.merge(trades[["trade_id", "entry_date"]], on="trade_id", how="left")
    daily = merged.groupby("date").apply(lambda g: float((g["mark_return"] * stake).sum()), include_groups=False).rename("unrealized_pnl").to_frame()
    realized = trades.assign(pnl=trades["net_return"] * stake).groupby("exit_date")["pnl"].sum()
    daily["realized_pnl"] = realized.reindex(daily.index).fillna(0.0)
    daily["equity"] = initial_capital + daily["realized_pnl"].cumsum() + daily["unrealized_pnl"]
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    ending = initial_capital + float((trades["net_return"] * stake).sum())
    return daily.reset_index(), {"initial_capital": initial_capital, "ending_equity": ending, "total_return": ending / initial_capital - 1.0, "max_drawdown": float(daily["drawdown"].min()), "trades": int(len(trades)), "worst_trade": float(trades["net_return"].min()), "mean_mae": float(trades["mae"].mean()), "mean_mfe": float(trades["mfe"].mean())}


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
    cfg = config.get("mark_to_market", {})
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in cfg.get("candidates", []):
        trades = build_rule_trades(events, spec["rule"], int(spec["horizon"]), config["validation"])
        paths, completed = trade_paths(trades, prices, cfg.get("stop_loss"))
        equity, summary = mark_to_market_portfolio(paths, completed, float(cfg.get("initial_capital", 100000)), float(cfg.get("position_fraction", 0.05)))
        name = spec["rule"] + f"_{spec['horizon']}d"
        paths.to_csv(out / f"mtm_paths_{name}.csv", index=False)
        completed.to_csv(out / f"mtm_trades_{name}.csv", index=False)
        equity.to_csv(out / f"mtm_equity_{name}.csv", index=False)
        rows.append({"rule": spec["rule"], "horizon": spec["horizon"], **summary, "research_status": "daily_path_tested", "production_approved": False})
    result = pd.DataFrame(rows)
    result.to_csv(out / "mark_to_market_summary.csv", index=False)
    print(result.to_string(index=False))
    print("\nProduction-approved candidates: 0")


if __name__ == "__main__":
    main()
