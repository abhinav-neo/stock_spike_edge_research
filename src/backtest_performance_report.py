from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator not in (0.0, -0.0) and np.isfinite(denominator) else np.nan


def calculate_performance_metrics(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
    initial_capital: float,
) -> dict:
    if equity.empty:
        return {
            "initial_capital": float(initial_capital),
            "final_equity": float(initial_capital),
            "net_profit": 0.0,
            "total_return": 0.0,
            "cagr": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "sortino": np.nan,
            "max_drawdown": 0.0,
            "calmar": np.nan,
            "trades": 0,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "expectancy": np.nan,
            "average_winner": np.nan,
            "average_loser": np.nan,
            "payoff_ratio": np.nan,
            "best_trade": np.nan,
            "worst_trade": np.nan,
            "average_holding_days": np.nan,
            "exposure": 0.0,
        }

    curve = equity.copy()
    curve["date"] = pd.to_datetime(curve["date"])
    curve = curve.sort_values("date").drop_duplicates("date", keep="last")
    if "daily_return" not in curve:
        curve["daily_return"] = curve["equity"].pct_change().fillna(0.0)
    if "drawdown" not in curve:
        curve["drawdown"] = curve["equity"] / curve["equity"].cummax() - 1.0

    start_date = pd.Timestamp(curve["date"].iloc[0])
    end_date = pd.Timestamp(curve["date"].iloc[-1])
    years = max((end_date - start_date).days / 365.25, 1.0 / 365.25)
    final_equity = float(curve["equity"].iloc[-1])
    total_return = final_equity / float(initial_capital) - 1.0
    cagr = (final_equity / float(initial_capital)) ** (1.0 / years) - 1.0 if final_equity > 0 else -1.0

    daily_returns = pd.to_numeric(curve["daily_return"], errors="coerce").dropna()
    daily_std = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else np.nan
    annualized_volatility = daily_std * math.sqrt(252) if np.isfinite(daily_std) else np.nan
    sharpe = _safe_ratio(float(daily_returns.mean()) * math.sqrt(252), daily_std)
    downside = daily_returns[daily_returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else np.nan
    sortino = _safe_ratio(float(daily_returns.mean()) * math.sqrt(252), downside_std)
    max_drawdown = float(-pd.to_numeric(curve["drawdown"], errors="coerce").min())
    calmar = _safe_ratio(cagr, max_drawdown)

    completed = trades.copy()
    if not completed.empty:
        completed["entry_date"] = pd.to_datetime(completed["entry_date"])
        completed["exit_date"] = pd.to_datetime(completed["exit_date"])
        returns = pd.to_numeric(completed["net_return"], errors="coerce").dropna()
        winners = returns[returns > 0]
        losers = returns[returns < 0]
        gross_profit = float(winners.sum())
        gross_loss = float(-losers.sum())
        average_winner = float(winners.mean()) if len(winners) else np.nan
        average_loser = float(losers.mean()) if len(losers) else np.nan
        holding_days = (completed["exit_date"] - completed["entry_date"]).dt.days

        active_dates: set[pd.Timestamp] = set()
        for _, trade in completed.iterrows():
            active_dates.update(pd.date_range(trade["entry_date"], trade["exit_date"], freq="B"))
        exposure = min(len(active_dates) / max(len(curve), 1), 1.0)
    else:
        returns = pd.Series(dtype=float)
        winners = returns
        losers = returns
        gross_profit = gross_loss = 0.0
        average_winner = average_loser = np.nan
        holding_days = pd.Series(dtype=float)
        exposure = 0.0

    return {
        "backtest_start": start_date.date().isoformat(),
        "backtest_end": end_date.date().isoformat(),
        "initial_capital": float(initial_capital),
        "final_equity": final_equity,
        "net_profit": final_equity - float(initial_capital),
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "calmar": calmar,
        "trades": int(len(returns)),
        "winning_trades": int((returns > 0).sum()),
        "losing_trades": int((returns < 0).sum()),
        "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
        "profit_factor": _safe_ratio(gross_profit, gross_loss),
        "expectancy": float(returns.mean()) if len(returns) else np.nan,
        "average_winner": average_winner,
        "average_loser": average_loser,
        "payoff_ratio": _safe_ratio(average_winner, abs(average_loser)) if np.isfinite(average_loser) else np.nan,
        "best_trade": float(returns.max()) if len(returns) else np.nan,
        "worst_trade": float(returns.min()) if len(returns) else np.nan,
        "average_holding_days": float(holding_days.mean()) if len(holding_days) else np.nan,
        "exposure": float(exposure),
    }


def period_returns(equity: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame(columns=["period", "return"])
    curve = equity.copy()
    curve["date"] = pd.to_datetime(curve["date"])
    curve = curve.sort_values("date").set_index("date")
    ending = curve["equity"].resample(frequency).last().dropna()
    returns = ending.pct_change()
    if len(returns):
        first_start = float(curve["equity"].iloc[0])
        returns.iloc[0] = float(ending.iloc[0]) / first_start - 1.0
    return pd.DataFrame({"period": ending.index.astype(str), "return": returns.values})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate comprehensive metrics from the historical portfolio backtest.")
    parser.add_argument("--trades", default="reports/alpha_portfolio_trades.csv")
    parser.add_argument("--equity", default="reports/alpha_portfolio_equity.csv")
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    trades_path = Path(args.trades)
    equity_path = Path(args.equity)
    if not trades_path.exists() or not equity_path.exists():
        raise FileNotFoundError("Run python -B -m src.alpha_portfolio before generating the performance report.")

    trades = pd.read_csv(trades_path)
    equity = pd.read_csv(equity_path)
    metrics = calculate_performance_metrics(trades, equity, args.initial_capital)
    annual = period_returns(equity, "YE")
    monthly = period_returns(equity, "ME")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(output / "backtest_performance_summary.csv", index=False)
    annual.to_csv(output / "backtest_annual_returns.csv", index=False)
    monthly.to_csv(output / "backtest_monthly_returns.csv", index=False)

    display = pd.DataFrame([metrics]).T.rename(columns={0: "value"})
    print(display.to_string())


if __name__ == "__main__":
    main()
