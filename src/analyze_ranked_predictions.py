from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def safe_correlation(actual: pd.Series, predicted: pd.Series) -> float:
    valid = actual.notna() & predicted.notna()
    if valid.sum() < 2:
        return np.nan
    a = actual.loc[valid].to_numpy(float)
    p = predicted.loc[valid].to_numpy(float)
    if np.std(a) == 0 or np.std(p) == 0:
        return np.nan
    return float(np.corrcoef(a, p)[0, 1])


def bucket_metrics(frame: pd.DataFrame, label: str) -> dict:
    actual = frame["actual_return"].astype(float)
    predicted = frame["predicted_return"].astype(float)
    return {
        "group": label,
        "trades": int(len(frame)),
        "avg_prediction": float(predicted.mean()) if len(frame) else np.nan,
        "avg_actual_return": float(actual.mean()) if len(frame) else np.nan,
        "median_actual_return": float(actual.median()) if len(frame) else np.nan,
        "return_std": float(actual.std(ddof=1)) if len(frame) > 1 else np.nan,
        "win_rate": float((actual > 0).mean()) if len(frame) else np.nan,
        "correlation": safe_correlation(actual, predicted),
        "minimum_return": float(actual.min()) if len(frame) else np.nan,
        "maximum_return": float(actual.max()) if len(frame) else np.nan,
    }


def percentile_report(frame: pd.DataFrame, percentiles: list[float]) -> pd.DataFrame:
    ordered = frame.sort_values("predicted_return", ascending=False).reset_index(drop=True)
    rows: list[dict] = []
    for pct in percentiles:
        count = max(1, int(np.floor(len(ordered) * pct)))
        rows.append(bucket_metrics(ordered.head(count), f"top_{pct:g}"))
        bottom = ordered.tail(count).copy()
        metrics = bucket_metrics(bottom, f"bottom_{pct:g}")
        metrics["short_avg_return"] = -metrics["avg_actual_return"]
        metrics["short_win_rate"] = float((bottom["actual_return"] < 0).mean())
        rows.append(metrics)
    rows.append(bucket_metrics(ordered, "all"))
    return pd.DataFrame(rows)


def decile_report(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.sort_values("predicted_return").copy()
    ranked["decile"] = pd.qcut(
        ranked["predicted_return"].rank(method="first"),
        10,
        labels=False,
    ) + 1
    rows = []
    for decile, group in ranked.groupby("decile", observed=True):
        metrics = bucket_metrics(group, f"decile_{int(decile)}")
        metrics["decile"] = int(decile)
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("decile")


def yearly_report(frame: pd.DataFrame, top_fraction: float) -> pd.DataFrame:
    work = frame.copy()
    work["year"] = pd.to_datetime(work["event_date"]).dt.year
    rows = []
    for year, group in work.groupby("year"):
        ordered = group.sort_values("predicted_return", ascending=False)
        count = max(1, int(np.floor(len(ordered) * top_fraction)))
        top = ordered.head(count)
        bottom = ordered.tail(count)
        rows.append(
            {
                "year": int(year),
                "events": int(len(group)),
                "correlation": safe_correlation(group["actual_return"], group["predicted_return"]),
                "all_avg_return": float(group["actual_return"].mean()),
                "top_trades": int(len(top)),
                "top_avg_return": float(top["actual_return"].mean()),
                "top_win_rate": float((top["actual_return"] > 0).mean()),
                "bottom_trades": int(len(bottom)),
                "bottom_avg_return": float(bottom["actual_return"].mean()),
                "bottom_short_avg_return": float(-bottom["actual_return"].mean()),
                "bottom_short_win_rate": float((bottom["actual_return"] < 0).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("year")


def simulate_ranked_portfolio(
    frame: pd.DataFrame,
    side: str,
    selection_fraction: float,
    holding_days: int,
    initial_capital: float,
    max_positions: int,
    cost_bps_round_trip: float,
    borrow_bps_per_day: float,
) -> tuple[pd.DataFrame, dict]:
    work = frame.copy()
    work["event_date"] = pd.to_datetime(work["event_date"])
    work = work.sort_values(["event_date", "predicted_return"], ascending=[True, side == "short"])

    selected_groups = []
    for _, group in work.groupby("event_date"):
        count = max(1, int(np.ceil(len(group) * selection_fraction)))
        chosen = group.nlargest(count, "predicted_return") if side == "long" else group.nsmallest(count, "predicted_return")
        selected_groups.append(chosen)
    selected = pd.concat(selected_groups, ignore_index=True) if selected_groups else work.iloc[0:0].copy()
    selected = selected.sort_values(["event_date", "predicted_return"], ascending=[True, side == "short"])

    capital = float(initial_capital)
    open_positions: list[dict] = []
    trades: list[dict] = []
    cost_rate = cost_bps_round_trip / 10_000.0
    borrow_rate = borrow_bps_per_day * holding_days / 10_000.0 if side == "short" else 0.0

    for row in selected.itertuples(index=False):
        entry_date = pd.Timestamp(row.event_date)
        still_open = []
        for position in open_positions:
            if position["exit_date"] <= entry_date:
                capital += position["pnl"]
            else:
                still_open.append(position)
        open_positions = still_open

        if len(open_positions) >= max_positions:
            continue

        slots_available = max_positions - len(open_positions)
        notional = capital / max(slots_available, 1)
        raw_return = float(row.actual_return) if side == "long" else -float(row.actual_return)
        net_return = raw_return - cost_rate - borrow_rate
        pnl = notional * net_return
        exit_date = entry_date + pd.offsets.BDay(holding_days)
        position = {"exit_date": exit_date, "pnl": pnl}
        open_positions.append(position)
        trades.append(
            {
                "symbol": row.symbol,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "side": side,
                "predicted_return": float(row.predicted_return),
                "actual_return": float(row.actual_return),
                "gross_strategy_return": raw_return,
                "net_strategy_return": net_return,
                "notional": notional,
                "pnl": pnl,
                "capital_before_entry": capital,
            }
        )

    for position in sorted(open_positions, key=lambda x: x["exit_date"]):
        capital += position["pnl"]

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return trades_df, {"side": side, "trades": 0, "ending_capital": initial_capital}

    trades_df = trades_df.sort_values("exit_date")
    trades_df["cumulative_pnl"] = trades_df["pnl"].cumsum()
    trades_df["equity"] = initial_capital + trades_df["cumulative_pnl"]
    trades_df["equity_peak"] = trades_df["equity"].cummax()
    trades_df["drawdown"] = trades_df["equity"] / trades_df["equity_peak"] - 1.0

    elapsed_days = max((trades_df["exit_date"].max() - trades_df["entry_date"].min()).days, 1)
    years = elapsed_days / 365.25
    ending_capital = float(trades_df["equity"].iloc[-1])
    cagr = (ending_capital / initial_capital) ** (1 / years) - 1 if ending_capital > 0 else -1.0
    returns = trades_df["net_strategy_return"]
    sharpe = float(np.sqrt(252 / holding_days) * returns.mean() / returns.std(ddof=1)) if len(returns) > 1 and returns.std(ddof=1) > 0 else np.nan

    summary = {
        "side": side,
        "selection_fraction": selection_fraction,
        "holding_days": holding_days,
        "trades": int(len(trades_df)),
        "initial_capital": initial_capital,
        "ending_capital": ending_capital,
        "total_return": ending_capital / initial_capital - 1,
        "cagr": float(cagr),
        "max_drawdown": float(trades_df["drawdown"].min()),
        "avg_net_trade_return": float(returns.mean()),
        "median_net_trade_return": float(returns.median()),
        "win_rate": float((returns > 0).mean()),
        "trade_level_sharpe_proxy": sharpe,
        "cost_bps_round_trip": cost_bps_round_trip,
        "borrow_bps_per_day": borrow_bps_per_day,
        "max_positions": max_positions,
        "warning": "This is an event-return approximation. It does not model intraday fills, halts, locate failures, margin calls, or mark-to-market paths between entry and exit.",
    }
    return trades_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ranked V5 model predictions")
    parser.add_argument("--input", required=True, help="Predictions CSV from train_predictive_model")
    parser.add_argument("--target", default="forward_return_5d")
    parser.add_argument("--period", default="test")
    parser.add_argument("--output-dir", default="reports/v5_rank_analysis")
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--holding-days", type=int, default=5)
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--borrow-bps-per-day", type=float, default=10.0)
    parser.add_argument("--skip-portfolio", action="store_true")
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    required = {"symbol", "event_date", args.target, "predicted_return", "period"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = source.loc[source["period"] == args.period, ["symbol", "event_date", args.target, "predicted_return"]].copy()
    frame = frame.rename(columns={args.target: "actual_return"})
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["actual_return", "predicted_return"])
    if len(frame) < 20:
        raise ValueError(f"Only {len(frame)} valid rows remain for period={args.period}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    deciles = decile_report(frame)
    percentiles = percentile_report(frame, [0.01, 0.02, 0.05, 0.10, 0.20])
    yearly = yearly_report(frame, args.top_fraction)
    deciles.to_csv(output / "decile_metrics.csv", index=False)
    percentiles.to_csv(output / "percentile_metrics.csv", index=False)
    yearly.to_csv(output / "yearly_stability.csv", index=False)

    summaries = {}
    if args.skip_portfolio:
        for side in ("long", "short"):
            stale = output / f"{side}_portfolio_trades.csv"
            if stale.exists():
                stale.unlink()
    else:
        for side in ("long", "short"):
            trades, summary = simulate_ranked_portfolio(
                frame=frame,
                side=side,
                selection_fraction=args.top_fraction,
                holding_days=args.holding_days,
                initial_capital=args.initial_capital,
                max_positions=args.max_positions,
                cost_bps_round_trip=args.cost_bps,
                borrow_bps_per_day=args.borrow_bps_per_day,
            )
            trades.to_csv(output / f"{side}_portfolio_trades.csv", index=False)
            summaries[side] = summary

    report = {
        "input": args.input,
        "period": args.period,
        "target": args.target,
        "rows_analyzed": int(len(frame)),
        "overall_correlation": safe_correlation(frame["actual_return"], frame["predicted_return"]),
        "portfolio_summaries": summaries,
    }
    (output / "analysis_summary.json").write_text(json.dumps(report, indent=2, default=str))

    print("\nDecile metrics")
    print(deciles[["decile", "trades", "avg_prediction", "avg_actual_return", "win_rate"]].to_string(index=False))
    print("\nPercentile metrics")
    print(percentiles[["group", "trades", "avg_actual_return", "median_actual_return", "win_rate"]].to_string(index=False))
    print("\nYearly stability")
    print(yearly.to_string(index=False))
    if not args.skip_portfolio:
        print("\nPortfolio summaries")
        print(json.dumps(summaries, indent=2, default=str))
    print(f"\nWrote ranked analysis to {output}")


if __name__ == "__main__":
    main()
