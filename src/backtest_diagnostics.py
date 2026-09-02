from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def grouped_performance(trades: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    columns = group_columns + [
        "trades", "win_rate", "mean_return", "median_return", "profit_factor",
        "average_winner", "average_loser", "best_trade", "worst_trade",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)

    frame = trades.copy()
    frame["net_return"] = pd.to_numeric(frame["net_return"], errors="coerce")
    frame = frame.dropna(subset=["net_return"])
    rows: list[dict] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        returns = group["net_return"]
        winners = returns[returns > 0]
        losers = returns[returns < 0]
        gross_profit = float(winners.sum())
        gross_loss = float(-losers.sum())
        row = dict(zip(group_columns, keys))
        row.update({
            "trades": int(len(returns)),
            "win_rate": float((returns > 0).mean()),
            "mean_return": float(returns.mean()),
            "median_return": float(returns.median()),
            "profit_factor": gross_profit / gross_loss if gross_loss > 0 else np.nan,
            "average_winner": float(winners.mean()) if len(winners) else np.nan,
            "average_loser": float(losers.mean()) if len(losers) else np.nan,
            "best_trade": float(returns.max()),
            "worst_trade": float(returns.min()),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mean_return", "trades"], ascending=[False, False]).reset_index(drop=True)


def cost_stress_test(trades: pd.DataFrame, extra_cost_bps: list[float]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["extra_cost_bps", "mean_net_return", "win_rate", "profit_factor", "profitable"])
    base = pd.to_numeric(trades["net_return"], errors="coerce").dropna()
    rows = []
    for bps in extra_cost_bps:
        stressed = base - float(bps) / 10000.0
        winners = stressed[stressed > 0]
        losers = stressed[stressed < 0]
        gross_loss = float(-losers.sum())
        profit_factor = float(winners.sum()) / gross_loss if gross_loss > 0 else np.nan
        rows.append({
            "extra_cost_bps": float(bps),
            "mean_net_return": float(stressed.mean()),
            "win_rate": float((stressed > 0).mean()),
            "profit_factor": profit_factor,
            "profitable": bool(float(stressed.mean()) > 0 and (np.isnan(profit_factor) or profit_factor > 1.0)),
        })
    return pd.DataFrame(rows)


def benchmark_comparison(equity: pd.DataFrame, prices: pd.DataFrame, symbol: str = "SPY") -> pd.DataFrame:
    columns = ["series", "start", "end", "total_return", "cagr", "max_drawdown", "annualized_volatility"]
    if equity.empty or prices.empty:
        return pd.DataFrame(columns=columns)

    curve = equity.copy()
    curve["date"] = pd.to_datetime(curve["date"])
    curve = curve.sort_values("date").drop_duplicates("date", keep="last")
    benchmark = prices[prices["symbol"].astype(str).str.upper().eq(symbol.upper())].copy()
    if benchmark.empty:
        return pd.DataFrame(columns=columns)
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark = benchmark.sort_values("date").drop_duplicates("date", keep="last")
    benchmark = benchmark[benchmark["date"].between(curve["date"].min(), curve["date"].max())]
    if len(benchmark) < 2:
        return pd.DataFrame(columns=columns)

    def summarize(name: str, dates: pd.Series, values: pd.Series) -> dict:
        values = pd.to_numeric(values, errors="coerce").dropna()
        aligned_dates = pd.to_datetime(dates).iloc[-len(values):]
        start = pd.Timestamp(aligned_dates.iloc[0])
        end = pd.Timestamp(aligned_dates.iloc[-1])
        years = max((end - start).days / 365.25, 1.0 / 365.25)
        total_return = float(values.iloc[-1] / values.iloc[0] - 1.0)
        cagr = float((values.iloc[-1] / values.iloc[0]) ** (1.0 / years) - 1.0)
        daily_returns = values.pct_change().dropna()
        drawdown = values / values.cummax() - 1.0
        return {
            "series": name,
            "start": start.date().isoformat(),
            "end": end.date().isoformat(),
            "total_return": total_return,
            "cagr": cagr,
            "max_drawdown": float(-drawdown.min()),
            "annualized_volatility": float(daily_returns.std(ddof=1) * np.sqrt(252)) if len(daily_returns) > 1 else np.nan,
        }

    strategy = summarize("strategy", curve["date"], curve["equity"])
    buy_hold = summarize(symbol.upper(), benchmark["date"], benchmark["close"])
    return pd.DataFrame([strategy, buy_hold], columns=columns)


def worst_trade_diagnostics(trades: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    frame = trades.copy()
    frame["net_return"] = pd.to_numeric(frame["net_return"], errors="coerce")
    preferred = [
        "symbol", "family", "direction", "signal_date", "entry_date", "exit_date",
        "entry_price", "exit_price", "gross_return", "net_return", "exit_reason",
        "stop_fill_type", "horizon", "candidate_rank",
    ]
    available = [column for column in preferred if column in frame.columns]
    return frame.sort_values("net_return").head(int(limit))[available].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose historical backtest weaknesses and robustness.")
    parser.add_argument("--trades", default="reports/alpha_portfolio_trades.csv")
    parser.add_argument("--equity", default="reports/alpha_portfolio_equity.csv")
    parser.add_argument("--prices", default="data/processed/daily_prices.parquet")
    parser.add_argument("--benchmark", default="SPY")
    parser.add_argument("--output-dir", default="reports/diagnostics")
    args = parser.parse_args()

    trades_path = Path(args.trades)
    equity_path = Path(args.equity)
    if not trades_path.exists() or not equity_path.exists():
        raise FileNotFoundError("Run python -B -m src.alpha_portfolio before diagnostics.")

    trades = pd.read_csv(trades_path)
    equity = pd.read_csv(equity_path)
    prices = pd.read_parquet(args.prices) if Path(args.prices).exists() else pd.DataFrame()
    if "exit_date" in trades:
        trades["exit_date"] = pd.to_datetime(trades["exit_date"])
        trades["exit_year"] = trades["exit_date"].dt.year

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    worst = worst_trade_diagnostics(trades)
    by_family = grouped_performance(trades, ["family"])
    by_direction = grouped_performance(trades, ["direction"])
    by_family_direction = grouped_performance(trades, ["family", "direction"])
    by_year = grouped_performance(trades, ["exit_year"]) if "exit_year" in trades else pd.DataFrame()
    stress = cost_stress_test(trades, [0, 5, 10, 25, 50, 100, 200])
    benchmark = benchmark_comparison(equity, prices, args.benchmark)

    worst.to_csv(output / "worst_trades.csv", index=False)
    by_family.to_csv(output / "performance_by_family.csv", index=False)
    by_direction.to_csv(output / "performance_by_direction.csv", index=False)
    by_family_direction.to_csv(output / "performance_by_family_direction.csv", index=False)
    by_year.to_csv(output / "performance_by_year.csv", index=False)
    stress.to_csv(output / "cost_stress_test.csv", index=False)
    benchmark.to_csv(output / "benchmark_comparison.csv", index=False)

    print("\nWorst trades\n", worst.head(10).to_string(index=False))
    print("\nPerformance by family/direction\n", by_family_direction.to_string(index=False))
    print("\nPerformance by year\n", by_year.to_string(index=False))
    print("\nExtra-cost stress test\n", stress.to_string(index=False))
    if benchmark.empty:
        print(f"\nBenchmark {args.benchmark.upper()} was not found in the price dataset.")
    else:
        print("\nBenchmark comparison\n", benchmark.to_string(index=False))


if __name__ == "__main__":
    main()
