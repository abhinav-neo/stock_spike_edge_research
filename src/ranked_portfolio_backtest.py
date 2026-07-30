from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TRADEABILITY_COLUMNS = {
    "min_price": ["entry_price", "price", "close"],
    "min_dollar_volume": ["avg_dollar_volume_20d", "dollar_volume_20d", "dollar_volume"],
    "min_market_cap": ["market_cap"],
}


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if suffix == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported file type for {source}; use CSV or Parquet")


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


def merge_tradeability_data(predictions: pd.DataFrame, features_input: str | None) -> tuple[pd.DataFrame, dict]:
    diagnostics: dict[str, object] = {"features_input": features_input}
    if not features_input:
        return predictions, diagnostics

    features = read_table(features_input)
    required_keys = {"symbol", "event_date"}
    missing = required_keys.difference(features.columns)
    if missing:
        raise ValueError(f"Features input missing merge keys: {sorted(missing)}")

    features = features.copy()
    features["event_date"] = pd.to_datetime(features["event_date"])
    available = sorted({column for columns in TRADEABILITY_COLUMNS.values() for column in columns if column in features.columns})
    if not available:
        raise ValueError(
            "Features input contains no supported tradeability columns. "
            f"Available columns include: {sorted(features.columns.tolist())}"
        )

    metadata = features[["symbol", "event_date", *available]].copy()
    duplicate_keys = int(metadata.duplicated(["symbol", "event_date"]).sum())
    if duplicate_keys:
        metadata = metadata.sort_values(["symbol", "event_date"]).drop_duplicates(
            ["symbol", "event_date"], keep="last"
        )

    existing = [column for column in available if column in predictions.columns]
    if existing:
        predictions = predictions.drop(columns=existing)
    merged = predictions.merge(metadata, on=["symbol", "event_date"], how="left", validate="many_to_one")
    diagnostics.update({
        "tradeability_columns_merged": available,
        "feature_duplicate_keys_removed": duplicate_keys,
        "rows_without_feature_match": int(merged[available].isna().all(axis=1).sum()),
    })
    return merged, diagnostics


def apply_tradeability_filters(
    frame: pd.DataFrame,
    min_price: float | None,
    min_dollar_volume: float | None,
    min_market_cap: float | None,
) -> tuple[pd.DataFrame, dict]:
    filtered = frame.copy()
    diagnostics: dict[str, object] = {"rows_before_tradeability_filters": int(len(filtered))}
    requested = {
        "min_price": min_price,
        "min_dollar_volume": min_dollar_volume,
        "min_market_cap": min_market_cap,
    }
    for label, minimum in requested.items():
        diagnostics[label] = minimum
        if minimum is None:
            continue
        candidates = TRADEABILITY_COLUMNS[label]
        column = next((name for name in candidates if name in filtered.columns), None)
        if column is None:
            raise ValueError(
                f"{label} requested but no supported column found. Tried: {candidates}. "
                "Pass --features-input data/processed/events_features_v5.parquet to merge metadata."
            )
        before = len(filtered)
        values = pd.to_numeric(filtered[column], errors="coerce")
        filtered = filtered.loc[values >= minimum].copy()
        diagnostics[f"{label}_column"] = column
        diagnostics[f"removed_by_{label}"] = int(before - len(filtered))
    diagnostics["rows_after_tradeability_filters"] = int(len(filtered))
    return filtered, diagnostics


def select_candidates(frame: pd.DataFrame, side: str, threshold: float) -> pd.DataFrame:
    if side == "long":
        selected = frame.loc[frame["predicted_return"] >= threshold].copy()
        ascending_prediction = False
    else:
        selected = frame.loc[frame["predicted_return"] <= threshold].copy()
        ascending_prediction = True
    return selected.sort_values(["event_date", "predicted_return"], ascending=[True, ascending_prediction])


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
    prevent_symbol_overlap: bool,
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
    skipped_capacity = skipped_symbol_overlap = skipped_cash = 0
    max_concurrent_positions = position_days = 0

    def close_positions(up_to_date: pd.Timestamp) -> None:
        nonlocal cash, realized_equity, open_positions
        closing = [position for position in open_positions if position["exit_date"] <= up_to_date]
        remaining = [position for position in open_positions if position["exit_date"] > up_to_date]
        for position in sorted(closing, key=lambda item: item["exit_date"]):
            cash += position["notional"] + position["pnl"]
            realized_equity += position["pnl"]
            equity_events.append({
                "date": position["exit_date"], "equity": realized_equity,
                "cash": cash, "open_positions": len(remaining),
            })
        open_positions = remaining

    for row in candidates.itertuples(index=False):
        entry_date = pd.Timestamp(row.event_date)
        close_positions(entry_date)
        open_symbols = {str(position["symbol"]) for position in open_positions}
        if prevent_symbol_overlap and str(row.symbol) in open_symbols:
            skipped_symbol_overlap += 1
            continue
        if len(open_positions) >= max_positions:
            skipped_capacity += 1
            continue
        if cash <= 0:
            skipped_cash += 1
            continue
        notional = min(allocation_per_position, cash)
        if notional <= 0:
            skipped_cash += 1
            continue
        cash -= notional
        gross_return = float(row.actual_return) if side == "long" else -float(row.actual_return)
        net_return = gross_return - cost_rate - borrow_rate
        pnl = notional * net_return
        exit_date = entry_date + pd.offsets.BDay(holding_days)
        position = {
            "symbol": row.symbol, "entry_date": entry_date, "exit_date": exit_date,
            "notional": notional, "pnl": pnl,
        }
        open_positions.append(position)
        max_concurrent_positions = max(max_concurrent_positions, len(open_positions))
        position_days += holding_days
        trade = {
            **position, "side": side, "predicted_return": float(row.predicted_return),
            "actual_return": float(row.actual_return), "gross_strategy_return": gross_return,
            "net_strategy_return": net_return, "cash_after_entry": cash,
        }
        for columns in TRADEABILITY_COLUMNS.values():
            for column in columns:
                if hasattr(row, column):
                    trade[column] = getattr(row, column)
        trades.append(trade)

    if open_positions:
        close_positions(max(position["exit_date"] for position in open_positions))

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_events)
    base_summary = {
        "side": side, "selection_fraction": fraction, "threshold_period": threshold_period,
        "prediction_threshold": threshold, "eligible_candidates": int(len(candidates)),
        "holding_days": holding_days, "initial_capital": initial_capital,
        "max_positions": max_positions, "allocation_per_position": allocation_per_position,
        "cost_bps_round_trip": cost_bps, "borrow_bps_per_day": borrow_bps_per_day,
        "prevent_symbol_overlap": prevent_symbol_overlap,
        "skipped_due_to_symbol_overlap": skipped_symbol_overlap,
        "skipped_due_to_capacity": skipped_capacity, "skipped_due_to_cash": skipped_cash,
        "max_concurrent_positions_observed": max_concurrent_positions,
    }
    if trades_df.empty:
        return trades_df, equity_df, {
            **base_summary, "trades": 0, "ending_capital": initial_capital, "total_return": 0.0,
            "warning": "No evaluation predictions crossed the calibrated threshold after constraints.",
        }

    equity_df = equity_df.sort_values("date").drop_duplicates("date", keep="last")
    equity_df["peak"] = equity_df["equity"].cummax()
    equity_df["drawdown"] = equity_df["equity"] / equity_df["peak"] - 1.0
    returns = trades_df["net_strategy_return"]
    pnl = trades_df["pnl"]
    start_date, end_date = trades_df["entry_date"].min(), trades_df["exit_date"].max()
    calendar_days = max((end_date - start_date).days, 1)
    years = max(calendar_days / 365.25, 1 / 365.25)
    ending_capital = float(realized_equity)
    cagr = (ending_capital / initial_capital) ** (1 / years) - 1 if ending_capital > 0 else -1.0
    std = returns.std(ddof=1)
    sharpe_proxy = float(np.sqrt(252 / holding_days) * returns.mean() / std) if len(returns) > 1 and std > 0 else np.nan
    downside = returns.loc[returns < 0]
    downside_std = downside.std(ddof=1)
    sortino_proxy = float(np.sqrt(252 / holding_days) * returns.mean() / downside_std) if len(downside) > 1 and downside_std > 0 else np.nan
    gross_profit = float(pnl.loc[pnl > 0].sum())
    gross_loss = float(-pnl.loc[pnl < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    max_drawdown = float(equity_df["drawdown"].min())
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else np.nan
    capital_utilization = min(1.0, position_days / max(calendar_days * max_positions * 252 / 365.25, 1))

    summary = {
        **base_summary, "trades": int(len(trades_df)), "unique_symbols": int(trades_df["symbol"].nunique()),
        "ending_capital": ending_capital, "total_return": ending_capital / initial_capital - 1,
        "cagr": float(cagr), "max_drawdown_on_realized_equity": max_drawdown,
        "calmar_ratio": float(calmar) if np.isfinite(calmar) else None,
        "average_net_trade_return": float(returns.mean()), "median_net_trade_return": float(returns.median()),
        "win_rate": float((returns > 0).mean()), "expectancy_dollars_per_trade": float(pnl.mean()),
        "gross_profit": gross_profit, "gross_loss": gross_loss,
        "profit_factor": float(profit_factor) if np.isfinite(profit_factor) else None,
        "trade_level_sharpe_proxy": sharpe_proxy, "trade_level_sortino_proxy": sortino_proxy,
        "approximate_capital_utilization": float(capital_utilization),
        "warning": (
            "Uses fixed-horizon returns and realized-equity drawdown; it does not model daily mark-to-market, "
            "halts, locate failures, margin calls, or changing borrow availability."
        ),
    }
    return trades_df, equity_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Capital-reserving ranked portfolio backtest")
    parser.add_argument("--input", required=True)
    parser.add_argument("--features-input", default=None, help="Optional CSV/Parquet metadata merged on symbol and event_date")
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
    parser.add_argument("--allow-symbol-overlap", action="store_true")
    parser.add_argument("--min-price", type=float, default=None)
    parser.add_argument("--min-dollar-volume", type=float, default=None)
    parser.add_argument("--min-market-cap", type=float, default=None)
    parser.add_argument("--output-dir", default="reports/v5_portfolio")
    args = parser.parse_args()

    source = read_table(args.input)
    required = {"symbol", "event_date", args.target, "predicted_return", "period"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    source["event_date"] = pd.to_datetime(source["event_date"])
    source = source.rename(columns={args.target: "actual_return"})
    source = source.replace([np.inf, -np.inf], np.nan).dropna(subset=["actual_return", "predicted_return"])
    source, merge_diagnostics = merge_tradeability_data(source, args.features_input)
    source, filter_diagnostics = apply_tradeability_filters(
        source, args.min_price, args.min_dollar_volume, args.min_market_cap
    )
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
            evaluation, side, threshold, args.fraction, args.threshold_period,
            args.holding_days, args.initial_capital, args.max_positions,
            args.cost_bps, args.borrow_bps_per_day, not args.allow_symbol_overlap,
        )
        trades.to_csv(output / f"{side}_trades.csv", index=False)
        equity.to_csv(output / f"{side}_equity.csv", index=False)
        summaries[side] = {**merge_diagnostics, **filter_diagnostics, **summary}

    (output / "portfolio_summary.json").write_text(json.dumps(summaries, indent=2, default=str))
    print(json.dumps(summaries, indent=2, default=str))
    print(f"Wrote portfolio outputs to {output}")


if __name__ == "__main__":
    main()
