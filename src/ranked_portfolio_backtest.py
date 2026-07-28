from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def score_threshold(source: pd.DataFrame, side: str, fraction: float, threshold_period: str) -> float:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    if "period" not in source.columns:
        raise ValueError("Prediction file must include period so thresholds can be calibrated without test leakage")
    calibration = source.loc[source["period"] == threshold_period, "predicted_return"].dropna().astype(float)
    if calibration.empty:
        raise ValueError(f"No predictions found for threshold-period={threshold_period}")
    quantile = 1.0 - fraction if side == "long" else fraction
    return float(calibration.quantile(quantile))


def select_candidates(frame: pd.DataFrame, side: str, threshold: float) -> pd.DataFrame:
    if side == "long":
        selected = frame.loc[frame["predicted_return"] >= threshold].copy()
        ascending_prediction = False
    else:
        selected = frame.loc[frame["predicted_return"] <= threshold].copy()
        ascending_prediction = True
    return selected.sort_values(
        ["event_date", "predicted_return"],
        ascending=[True, ascending_prediction],
    )


def simulate(
    frame: pd.DataFrame,
    side: str,
    threshold: float,
    fraction: float,
    threshold_period: str,
    holding_days: int,
    initial_capital: float,
    max_positions: int,
    cost_bps: float,
    borrow_bps_per_day: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    candidates = select_candidates(frame, side, threshold)
    cash = float(initial_capital)
    realized_equity = float(initial_capital)
    open_positions: list[dict] = []
    trades: list[dict] = []
    equity_events: list[dict] = []
    cost_rate = cost_bps / 10_000.0
    borrow_rate = holding_days * borrow_bps_per_day / 10_000.0 if side == "short" else 0.0
    allocation_per_position = initial_capital / max_positions

    def close_positions(up_to_date: pd.Timestamp) -> None:
        nonlocal cash, realized_equity, open_positions
        closing = [position for position in open_positions if position["exit_date"] <= up_to_date]
        remaining = [position for position in open_positions if position["exit_date"] > up_to_date]
        for position in sorted(closing, key=lambda item: item["exit_date"]):
            cash += position["notional"] + position["pnl"]
            realized_equity += position["pnl"]
            equity_events.append(
                {
                    "date": position["exit_date"],
                    "equity": realized_equity,
                    "cash": cash,
                    "open_positions": len(remaining),
                }
            )
        open_positions = remaining

    for row in candidates.itertuples(index=False):
        entry_date = pd.Timestamp(row.event_date)
        close_positions(entry_date)
        if len(open_positions) >= max_positions or cash <= 0:
            continue

        notional = min(allocation_per_position, cash)
        if notional <= 0:
            continue
        cash -= notional
        gross_return = float(row.actual_return) if side == "long" else -float(row.actual_return)
        net_return = gross_return - cost_rate - borrow_rate
        pnl = notional * net_return
        exit_date = entry_date + pd.offsets.BDay(holding_days)
        position = {
            "symbol": row.symbol,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "notional": notional,
            "pnl": pnl,
        }
        open_positions.append(position)
        trades.append(
            {
                **position,
                "side": side,
                "predicted_return": float(row.predicted_return),
                "actual_return": float(row.actual_return),
                "gross_strategy_return": gross_return,
                "net_strategy_return": net_return,
                "cash_after_entry": cash,
            }
        )

    if open_positions:
        close_positions(max(position["exit_date"] for position in open_positions))

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_events)
    base_summary = {
        "side": side,
        "selection_fraction": fraction,
        "threshold_period": threshold_period,
        "prediction_threshold": threshold,
        "eligible_candidates": int(len(candidates)),
        "holding_days": holding_days,
        "initial_capital": initial_capital,
        "max_positions": max_positions,
        "allocation_per_position": allocation_per_position,
        "cost_bps_round_trip": cost_bps,
        "borrow_bps_per_day": borrow_bps_per_day,
    }
    if trades_df.empty:
        return trades_df, equity_df, {
            **base_summary,
            "trades": 0,
            "ending_capital": initial_capital,
            "total_return": 0.0,
            "warning": "No test predictions crossed the validation-calibrated threshold.",
        }

    equity_df = equity_df.sort_values("date").drop_duplicates("date", keep="last")
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = equity_df["equity"] / equity_df["peak"] - 1.0
    returns = trades_df["net_strategy_return"]
    start_date = trades_df["entry_date"].min()
    end_date = trades_df["exit_date"].max()
    years = max((end_date - start_date).days / 365.25, 1 / 365.25)
    ending_capital = float(realized_equity)
    cagr = (ending_capital / initial_capital) ** (1 / years) - 1 if ending_capital > 0 else -1.0
    sharpe_proxy = (
        float(np.sqrt(252 / holding_days) * returns.mean() / returns.std(ddof=1))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else np.nan
    )
    summary = {
        **base_summary,
        "trades": int(len(trades_df)),
        "ending_capital": ending_capital,
        "total_return": ending_capital / initial_capital - 1,
        "cagr": float(cagr),
        "max_drawdown_on_realized_equity": float(equity_df["drawdown"].min()),
        "average_net_trade_return": float(returns.mean()),
        "median_net_trade_return": float(returns.median()),
        "win_rate": float((returns > 0).mean()),
        "trade_level_sharpe_proxy": sharpe_proxy,
        "warning": (
            "Uses fixed-horizon event returns and realized-equity drawdown. It does not model intraday fills, "
            "mark-to-market paths, halts, locate failures, margin calls, or changing borrow availability."
        ),
    }
    return trades_df, equity_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Capital-reserving ranked portfolio backtest")
    parser.add_argument("--input", required=True)
    parser.add_argument("--target", default="forward_return_5d")
    parser.add_argument("--period", default="test")
    parser.add_argument("--threshold-period", default="validation")
    parser.add_argument("--side", choices=["long", "short", "both"], default="both")
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--holding-days", type=int, default=5)
    parser.add_argument("--initial-capital", type=float, default=100000.0)
    parser.add_argument("--max-positions", type=int, default=10)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--borrow-bps-per-day", type=float, default=10.0)
    parser.add_argument("--output-dir", default="reports/v5_portfolio")
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    required = {"symbol", "event_date", args.target, "predicted_return", "period"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    source["event_date"] = pd.to_datetime(source["event_date"])
    source = source.rename(columns={args.target: "actual_return"})
    source = source.replace([np.inf, -np.inf], np.nan).dropna(subset=["actual_return", "predicted_return"])
    evaluation = source.loc[source["period"] == args.period].copy()
    if evaluation.empty:
        raise ValueError(f"No valid prediction rows remain for period={args.period}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sides = ["long", "short"] if args.side == "both" else [args.side]
    summaries = {}
    for side in sides:
        threshold = score_threshold(source, side, args.fraction, args.threshold_period)
        trades, equity, summary = simulate(
            evaluation,
            side,
            threshold,
            args.fraction,
            args.threshold_period,
            args.holding_days,
            args.initial_capital,
            args.max_positions,
            args.cost_bps,
            args.borrow_bps_per_day,
        )
        trades.to_csv(output / f"{side}_trades.csv", index=False)
        equity.to_csv(output / f"{side}_equity.csv", index=False)
        summaries[side] = summary

    (output / "portfolio_summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print(json.dumps(summaries, indent=2, default=str))
    print(f"Wrote portfolio outputs to {output}")


if __name__ == "__main__":
    main()
