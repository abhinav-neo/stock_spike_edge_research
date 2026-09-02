from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.mark_to_market import apply_locate_model
from src.ranked_portfolio_backtest import apply_tradeability_filters, join_feature_data, score_threshold, simulate
from src.v5_mtm_research import daily_equity_curve, load_candidate_prices, normalize_spy


def period_trades(predictions: pd.DataFrame, features: pd.DataFrame, period: str, side: str) -> pd.DataFrame:
    source = join_feature_data(predictions, features).rename(columns={"forward_return_5d": "actual_return"})
    source, _ = apply_tradeability_filters(source, 10.0, 5_000_000.0, None)
    threshold = score_threshold(source, side, 0.10, "validation")
    trades, _, _ = simulate(
        source.loc[source["period"] == period].copy(), side, threshold, 0.10,
        "validation", 5, 100_000.0, 100, 0.0, 0.0, True,
    )
    events = features[["symbol", "event_date", "entry_date"]].copy()
    events["entry_date"] = pd.to_datetime(events["entry_date"]).dt.normalize()
    events["event_date"] = pd.to_datetime(events["event_date"]).dt.normalize()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"]).dt.normalize()
    result = trades.merge(events, on=["symbol", "entry_date"], how="left", validate="many_to_one")
    if result["event_date"].isna().any():
        raise ValueError("Could not attach event dates to all overlay trades")
    result["side"] = side
    return result


def directional_paths(trades: pd.DataFrame, prices: pd.DataFrame, annual_borrow_bps: float, cost_bps: float, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    path_rows, completed_rows = [], []
    daily_borrow = annual_borrow_bps / 10_000.0 / 252.0
    cost = cost_bps / 10_000.0
    for sequence, trade in trades.reset_index(drop=True).iterrows():
        after = prices.loc[(prices["symbol"] == trade.symbol) & (prices["date"] > trade.event_date)].sort_values("date").head(5)
        if len(after) < 5:
            continue
        entry = float(after.iloc[0]["open"])
        direction = 1.0 if trade.side == "long" else -1.0
        trade_id = f"{prefix}-{sequence}"
        for day, row in enumerate(after.itertuples(index=False), start=1):
            gross = direction * (float(row.close) / entry - 1.0)
            borrow = daily_borrow * day if trade.side == "short" else 0.0
            path_rows.append({"trade_id": trade_id, "symbol": trade.symbol, "side": trade.side, "date": row.date, "day": day, "mark_return": gross - borrow - cost})
        exit_row = after.iloc[-1]
        gross = direction * (float(exit_row["close"]) / entry - 1.0)
        borrow = daily_borrow * 5 if trade.side == "short" else 0.0
        completed_rows.append({"trade_id": trade_id, "symbol": trade.symbol, "side": trade.side, "entry_date": after.iloc[0]["date"], "exit_date": exit_row["date"], "net_return": gross - borrow - cost})
    return pd.DataFrame(path_rows), pd.DataFrame(completed_rows)


def combined_metrics(paths: pd.DataFrame, trades: pd.DataFrame, spy: pd.DataFrame, fraction: float) -> tuple[dict, pd.DataFrame]:
    calendar = pd.DatetimeIndex(spy["date"])
    alpha = daily_equity_curve(paths, trades, calendar, 100_000.0, 100_000.0 * fraction)
    aligned = alpha.merge(spy, on="date", how="left").dropna(subset=["benchmark"]).copy()
    aligned["spy_equity"] = 100_000.0 * aligned["benchmark"] / aligned["benchmark"].iloc[0]
    max_positions = int(aligned["concurrent_positions"].max())
    core_weight = max(0.0, 1.0 - max_positions * fraction)
    aligned["combined_equity"] = 100_000.0 + core_weight * (aligned["spy_equity"] - 100_000.0) + (aligned["equity"] - 100_000.0)
    aligned["combined_return"] = aligned["combined_equity"].pct_change().fillna(0.0)
    aligned["combined_drawdown"] = aligned["combined_equity"] / aligned["combined_equity"].cummax() - 1.0
    years = max((aligned["date"].iloc[-1] - aligned["date"].iloc[0]).days / 365.25, 1 / 365.25)
    returns = aligned["combined_return"]
    cagr = (aligned["combined_equity"].iloc[-1] / 100_000.0) ** (1 / years) - 1.0
    spy_cagr = (aligned["spy_equity"].iloc[-1] / 100_000.0) ** (1 / years) - 1.0
    std = returns.std(ddof=1)
    spy_returns = aligned["spy_equity"].pct_change().fillna(0.0)
    return {
        "position_fraction": fraction,
        "core_weight": core_weight,
        "maximum_positions": max_positions,
        "maximum_gross_exposure": core_weight + max_positions * fraction,
        "trades": len(trades),
        "cagr": float(cagr),
        "spy_cagr": float(spy_cagr),
        "excess_cagr": float(cagr - spy_cagr),
        "max_drawdown": float(aligned["combined_drawdown"].min()),
        "sharpe_zero_rate": float(returns.mean() / std * np.sqrt(252)) if std > 0 else None,
        "spy_sharpe_zero_rate": float(spy_returns.mean() / spy_returns.std(ddof=1) * np.sqrt(252)),
        "total_return": float(aligned["combined_equity"].iloc[-1] / 100_000.0 - 1.0),
        "spy_total_return": float(aligned["spy_equity"].iloc[-1] / 100_000.0 - 1.0),
    }, aligned


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-sized, gross-capped SPY plus V5 long/short overlay")
    parser.add_argument("--predictions", default="reports/v5_validation_suite/model/predictions.csv")
    parser.add_argument("--features", default="data/processed/events_features_v5.parquet")
    parser.add_argument("--raw-prices-dir", default="data/raw/prices")
    parser.add_argument("--spy", default="data/raw/spy_benchmark.parquet")
    parser.add_argument("--output-dir", default="reports/v5_bounded_overlay")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    features = pd.read_parquet(args.features)
    spy = normalize_spy(Path(args.spy))
    sets = {}
    all_symbols = []
    for period in ("validation", "test"):
        longs = period_trades(predictions, features, period, "long")
        shorts = period_trades(predictions, features, period, "short")
        sets[period] = (longs, shorts)
        all_symbols.extend(longs["symbol"].tolist() + shorts["symbol"].tolist())
    prices = load_candidate_prices(pd.Series(all_symbols), Path(args.raw_prices_dir))

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    period_data = {}
    for period, (longs, shorts) in sets.items():
        located_shorts, rejected = apply_locate_model(shorts, 0.70, 42)
        long_paths, long_done = directional_paths(longs, prices, 0.0, 100.0, f"{period}-long")
        short_paths, short_done = directional_paths(located_shorts, prices, 1000.0, 100.0, f"{period}-short")
        period_data[period] = (pd.concat([long_paths, short_paths], ignore_index=True), pd.concat([long_done, short_done], ignore_index=True), len(rejected))

    validation_rows = []
    for fraction in (0.0, 0.005, 0.01, 0.015, 0.02):
        metrics, _ = combined_metrics(*period_data["validation"][:2], spy, fraction)
        validation_rows.append(metrics)
    validation = pd.DataFrame(validation_rows)
    eligible = validation.loc[(validation["max_drawdown"] >= -0.25) & (validation["excess_cagr"] > 0) & (validation["sharpe_zero_rate"] > validation["spy_sharpe_zero_rate"])]
    chosen = float(eligible.sort_values(["excess_cagr", "sharpe_zero_rate"], ascending=False).iloc[0]["position_fraction"]) if not eligible.empty else 0.0
    validation["selected"] = validation["position_fraction"].eq(chosen)
    validation.to_csv(output / "validation_sizing.csv", index=False)

    test_metrics, test_curve = combined_metrics(*period_data["test"][:2], spy, chosen)
    test_metrics.update({"validation_selected_fraction": chosen, "long_trades": int((period_data["test"][1]["side"] == "long").sum()), "short_trades": int((period_data["test"][1]["side"] == "short").sum()), "short_locate_rejections": period_data["test"][2], "accepted": bool(chosen > 0 and test_metrics["excess_cagr"] > 0 and test_metrics["sharpe_zero_rate"] > test_metrics["spy_sharpe_zero_rate"] and test_metrics["max_drawdown"] >= -0.25)})
    test_curve.to_csv(output / "test_daily_equity.csv", index=False)
    (output / "test_summary.json").write_text(json.dumps(test_metrics, indent=2))
    print(validation.to_string(index=False))
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
