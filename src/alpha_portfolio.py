from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.alpha_factory import build_features, candidate_mask


def row_to_spec(row: pd.Series) -> dict:
    spec = {"family": row["family"], "direction": row["direction"], "horizon": int(row["horizon"])}
    optional = ["gap", "rvol", "close_location", "return_threshold", "ma20_distance", "breakout_threshold"]
    for key in optional:
        if key in row.index and pd.notna(row[key]):
            spec[key] = float(row[key])
    return spec


def candidate_key(spec: dict) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def select_representatives(survivors: pd.DataFrame, maximum: int, per_family: int) -> pd.DataFrame:
    if survivors.empty:
        return survivors.copy()
    frame = survivors.copy()
    score_col = "test_cluster_t_stat" if "test_cluster_t_stat" in frame else "test_daily_mean"
    frame = frame.sort_values([score_col, "test_daily_mean"], ascending=False)
    selected = []
    family_counts: dict[tuple[str, str], int] = {}
    seen_shapes: set[tuple] = set()
    for _, row in frame.iterrows():
        group = (str(row["family"]), str(row["direction"]))
        if family_counts.get(group, 0) >= per_family:
            continue
        # Collapse near-clones to one representative per family/direction/horizon.
        shape = (group[0], group[1], int(row["horizon"]))
        if shape in seen_shapes:
            continue
        selected.append(row)
        seen_shapes.add(shape)
        family_counts[group] = family_counts.get(group, 0) + 1
        if len(selected) >= maximum:
            break
    return pd.DataFrame(selected).reset_index(drop=True)


def build_trades(features: pd.DataFrame, selected: pd.DataFrame, cfg: dict, start: str) -> pd.DataFrame:
    rows: list[dict] = []
    groups = {symbol: data.reset_index(drop=True) for symbol, data in features.groupby("symbol", sort=False)}
    for rank, (_, survivor) in enumerate(selected.iterrows(), start=1):
        spec = row_to_spec(survivor)
        mask = candidate_mask(features, spec, cfg) & features["date"].ge(pd.Timestamp(start))
        for idx in features.index[mask]:
            symbol = features.at[idx, "symbol"]
            signal_date = features.at[idx, "date"]
            local = groups[symbol]
            positions = local.index[local["date"].eq(signal_date)]
            if len(positions) != 1:
                continue
            signal_i = int(positions[0])
            entry_i = signal_i + 1
            exit_i = entry_i + int(spec["horizon"]) - 1
            if entry_i >= len(local) or exit_i >= len(local):
                continue
            rows.append({
                "candidate_rank": rank,
                "candidate_key": candidate_key(spec),
                "family": spec["family"],
                "direction": spec["direction"],
                "symbol": symbol,
                "signal_date": signal_date,
                "entry_date": local.at[entry_i, "date"],
                "scheduled_exit_date": local.at[exit_i, "date"],
                "entry_price": float(local.at[entry_i, "open"]),
                "horizon": int(spec["horizon"]),
            })
    if not rows:
        return pd.DataFrame()
    trades = pd.DataFrame(rows)
    # A stock can trigger several correlated candidates on the same day. Keep the highest-ranked one.
    return trades.sort_values("candidate_rank").drop_duplicates(["symbol", "entry_date", "direction"]).reset_index(drop=True)


def apply_capacity(trades: pd.DataFrame, max_daily_entries: int, max_concurrent: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return trades.copy(), trades.copy()
    accepted, rejected = [], []
    active_exits: list[pd.Timestamp] = []
    daily_counts: dict[pd.Timestamp, int] = {}
    for _, trade in trades.sort_values(["entry_date", "candidate_rank", "symbol"]).iterrows():
        entry = pd.Timestamp(trade["entry_date"])
        active_exits = [date for date in active_exits if date >= entry]
        reason = None
        if daily_counts.get(entry, 0) >= max_daily_entries:
            reason = "daily_entry_cap"
        elif len(active_exits) >= max_concurrent:
            reason = "concurrency_cap"
        if reason:
            item = trade.to_dict(); item["rejection_reason"] = reason; rejected.append(item)
        else:
            accepted.append(trade.to_dict())
            active_exits.append(pd.Timestamp(trade["scheduled_exit_date"]))
            daily_counts[entry] = daily_counts.get(entry, 0) + 1
    return pd.DataFrame(accepted), pd.DataFrame(rejected)


def execute_trade(trade: pd.Series, prices: pd.DataFrame, stop_loss: float | None, cost_bps: float) -> dict:
    symbol_prices = prices[(prices["symbol"] == trade["symbol"]) & prices["date"].between(trade["entry_date"], trade["scheduled_exit_date"])].sort_values("date")
    entry = float(trade["entry_price"])
    direction = trade["direction"]
    exit_price = float(symbol_prices.iloc[-1]["close"])
    exit_date = pd.Timestamp(symbol_prices.iloc[-1]["date"])
    exit_reason = "time"
    stop_fill_type = "none"
    if stop_loss is not None:
        stop = entry * (1.0 - stop_loss) if direction == "long" else entry * (1.0 + stop_loss)
        for _, bar in symbol_prices.iterrows():
            if direction == "long":
                if float(bar["open"]) <= stop:
                    exit_price, exit_date, exit_reason, stop_fill_type = float(bar["open"]), bar["date"], "stop", "gap_open"; break
                if float(bar["low"]) <= stop:
                    exit_price, exit_date, exit_reason, stop_fill_type = stop, bar["date"], "stop", "intraday"; break
            else:
                if float(bar["open"]) >= stop:
                    exit_price, exit_date, exit_reason, stop_fill_type = float(bar["open"]), bar["date"], "stop", "gap_open"; break
                if float(bar["high"]) >= stop:
                    exit_price, exit_date, exit_reason, stop_fill_type = stop, bar["date"], "stop", "intraday"; break
    gross = exit_price / entry - 1.0
    if direction == "short":
        gross = -gross
    net = gross - float(cost_bps) / 10000.0
    result = trade.to_dict()
    result.update({"exit_date": exit_date, "exit_price": exit_price, "gross_return": gross, "net_return": net,
                   "exit_reason": exit_reason, "stop_fill_type": stop_fill_type})
    return result


def portfolio_summary(trades: pd.DataFrame, initial_capital: float, position_fraction: float) -> tuple[pd.DataFrame, dict]:
    if trades.empty:
        return pd.DataFrame(), {"trades": 0, "total_return": 0.0, "cagr": np.nan, "max_drawdown": 0.0, "sharpe": np.nan}
    stake = initial_capital * position_fraction
    pnl = trades.assign(pnl=trades["net_return"] * stake).groupby("exit_date")["pnl"].sum().sort_index()
    dates = pd.date_range(trades["entry_date"].min(), trades["exit_date"].max(), freq="B")
    daily = pnl.reindex(dates, fill_value=0.0)
    equity = initial_capital + daily.cumsum()
    returns = equity.pct_change().fillna(0.0)
    peak = equity.cummax()
    drawdown = equity / peak - 1.0
    years = max((dates[-1] - dates[0]).days / 365.25, 1 / 365.25)
    ending = float(equity.iloc[-1])
    cagr = (ending / initial_capital) ** (1.0 / years) - 1.0 if ending > 0 else -1.0
    sharpe = float(returns.mean() / returns.std(ddof=1) * math.sqrt(252)) if returns.std(ddof=1) > 0 else np.nan
    curve = pd.DataFrame({"date": dates, "daily_pnl": daily.values, "equity": equity.values, "daily_return": returns.values, "drawdown": drawdown.values})
    return curve, {"trades": int(len(trades)), "total_return": ending / initial_capital - 1.0, "cagr": cagr,
                   "max_drawdown": float(-drawdown.min()), "sharpe": sharpe,
                   "win_rate": float((trades["net_return"] > 0).mean()), "worst_trade": float(trades["net_return"].min())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/alpha_factory.yaml")
    parser.add_argument("--prices", default="data/processed/daily_prices.parquet")
    parser.add_argument("--survivors", default="reports/alpha_factory_locked_test_survivors.csv")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    root = yaml.safe_load(Path(args.config).read_text())
    factory_cfg = root.get("alpha_factory", {})
    portfolio_cfg = root.get("alpha_portfolio", {})
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    survivors_path = Path(args.survivors)
    if not survivors_path.exists():
        raise FileNotFoundError(f"Run python -B -m src.validate_alpha_factory first; missing {survivors_path}")
    survivors = pd.read_csv(survivors_path)
    selected = select_representatives(survivors, int(portfolio_cfg.get("maximum_candidates", 12)), int(portfolio_cfg.get("maximum_per_family_direction", 2)))
    selected.to_csv(output / "alpha_portfolio_selected_candidates.csv", index=False)
    if selected.empty:
        pd.DataFrame().to_csv(output / "alpha_portfolio_summary.csv", index=False)
        print("Locked-test survivors: 0\nPortfolio simulation skipped.\nProduction-approved candidates: 0")
        return

    prices = pd.read_parquet(args.prices); prices["date"] = pd.to_datetime(prices["date"])
    features = build_features(prices)
    trades = build_trades(features, selected, factory_cfg, root.get("validation", {}).get("test_start", "2023-01-01"))
    accepted, rejected = apply_capacity(trades, int(portfolio_cfg.get("max_daily_entries", 3)), int(portfolio_cfg.get("max_concurrent_positions", 10)))
    completed = pd.DataFrame([execute_trade(row, prices, portfolio_cfg.get("stop_loss", 0.25), portfolio_cfg.get("round_trip_cost_bps", 100.0)) for _, row in accepted.iterrows()])
    curve, summary = portfolio_summary(completed, float(portfolio_cfg.get("initial_capital", 100000)), float(portfolio_cfg.get("position_fraction", 0.05)))
    summary.update({"selected_candidates": int(len(selected)), "candidate_trades": int(len(trades)), "capacity_rejections": int(len(rejected)),
                    "target_cagr": float(portfolio_cfg.get("target_cagr", 0.40)),
                    "target_met": bool(np.isfinite(summary["cagr"]) and summary["cagr"] >= float(portfolio_cfg.get("target_cagr", 0.40))),
                    "production_approved": False})
    completed.to_csv(output / "alpha_portfolio_trades.csv", index=False)
    rejected.to_csv(output / "alpha_portfolio_rejections.csv", index=False)
    curve.to_csv(output / "alpha_portfolio_equity.csv", index=False)
    pd.DataFrame([summary]).to_csv(output / "alpha_portfolio_summary.csv", index=False)
    print(pd.DataFrame([summary]).to_string(index=False))
    print("Production-approved candidates: 0")


if __name__ == "__main__":
    main()
