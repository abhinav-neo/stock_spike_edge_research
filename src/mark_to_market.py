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


def trade_paths(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    stop_loss: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct daily paths for short trades.

    A stop is gap-aware: if the session opens above the stop, the modeled cover
    occurs at the opening price. Otherwise, an intraday high breach fills at the
    stop price. This is still a daily-bar approximation and cannot model halts or
    intraday slippage.
    """
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
        stop_fill_type = "none"
        exit_price = float(after.iloc[-1]["close"])
        exit_date = after.iloc[-1]["date"]
        mae = 0.0
        mfe = 0.0
        stop_price = entry * (1.0 + float(stop_loss)) if stop_loss is not None else None

        for day_number, (_, row) in enumerate(after.iterrows(), start=1):
            day_open = float(row["open"])
            day_high = float(row["high"])
            day_low = float(row["low"])
            day_close = float(row["close"])

            adverse = (entry - day_high) / entry
            favorable = (entry - day_low) / entry
            mae = min(mae, adverse)
            mfe = max(mfe, favorable)

            stopped = False
            mark_price = day_close
            if stop_price is not None:
                if day_open >= stop_price:
                    stopped = True
                    mark_price = day_open
                    stop_fill_type = "gap_open"
                elif day_high >= stop_price:
                    stopped = True
                    mark_price = stop_price
                    stop_fill_type = "intraday_stop"

            mark_ret = (entry - mark_price) / entry
            path_rows.append(
                {
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "date": row["date"],
                    "day": day_number,
                    "mark_return": mark_ret,
                }
            )

            if stopped:
                exit_reason = "stop"
                exit_price = mark_price
                exit_date = row["date"]
                break

        net_return = (entry - exit_price) / entry
        trade_rows.append(
            {
                "trade_id": trade_id,
                "symbol": symbol,
                "entry_date": after.iloc[0]["date"],
                "exit_date": exit_date,
                "entry_price": entry,
                "exit_price": exit_price,
                "net_return": net_return,
                "mae": mae,
                "mfe": mfe,
                "exit_reason": exit_reason,
                "stop_fill_type": stop_fill_type,
            }
        )

    return pd.DataFrame(path_rows), pd.DataFrame(trade_rows)


def _risk_statistics(equity: pd.DataFrame) -> dict:
    if equity.empty or len(equity) < 2:
        return {"cagr": 0.0, "annualized_volatility": 0.0, "sharpe": np.nan, "sortino": np.nan, "calmar": np.nan}

    daily_returns = equity["equity"].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    elapsed_days = max((equity["date"].iloc[-1] - equity["date"].iloc[0]).days, 1)
    years = elapsed_days / 365.25
    start_equity = float(equity["equity"].iloc[0])
    end_equity = float(equity["equity"].iloc[-1])
    cagr = (end_equity / start_equity) ** (1.0 / years) - 1.0 if start_equity > 0 and end_equity > 0 else np.nan

    volatility = float(daily_returns.std(ddof=1) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0
    mean_daily = float(daily_returns.mean()) if len(daily_returns) else 0.0
    daily_std = float(daily_returns.std(ddof=1)) if len(daily_returns) > 1 else 0.0
    downside = daily_returns[daily_returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sharpe = mean_daily / daily_std * np.sqrt(252) if daily_std > 0 else np.nan
    sortino = mean_daily / downside_std * np.sqrt(252) if downside_std > 0 else np.nan
    max_drawdown = abs(float(equity["drawdown"].min()))
    calmar = cagr / max_drawdown if max_drawdown > 0 and np.isfinite(cagr) else np.nan
    return {
        "cagr": float(cagr),
        "annualized_volatility": volatility,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "sortino": float(sortino) if np.isfinite(sortino) else np.nan,
        "calmar": float(calmar) if np.isfinite(calmar) else np.nan,
    }


def mark_to_market_portfolio(
    paths: pd.DataFrame,
    trades: pd.DataFrame,
    initial_capital: float = 100000.0,
    position_fraction: float = 0.05,
) -> tuple[pd.DataFrame, dict]:
    if paths.empty or trades.empty:
        return pd.DataFrame(), {
            "initial_capital": initial_capital,
            "ending_equity": initial_capital,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "trades": 0,
        }

    stake = initial_capital * position_fraction
    paths = paths.copy()
    trades = trades.copy()
    paths["date"] = pd.to_datetime(paths["date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    trades["pnl"] = trades["net_return"] * stake

    dates = pd.Index(sorted(paths["date"].unique()), name="date")
    current_marks = paths.assign(mark_pnl=paths["mark_return"] * stake).groupby("date")["mark_pnl"].sum()
    realized_by_date = trades.groupby("exit_date")["pnl"].sum().reindex(dates, fill_value=0.0)
    realized_before_date = realized_by_date.cumsum().shift(1, fill_value=0.0)

    daily = pd.DataFrame(index=dates)
    daily["realized_pnl_prior"] = realized_before_date
    daily["active_mark_pnl"] = current_marks.reindex(dates, fill_value=0.0)
    daily["equity"] = initial_capital + daily["realized_pnl_prior"] + daily["active_mark_pnl"]
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    daily = daily.reset_index()

    ending = initial_capital + float(trades["pnl"].sum())
    risk = _risk_statistics(daily)
    stop_trades = trades[trades["exit_reason"] == "stop"] if "exit_reason" in trades else trades.iloc[0:0]
    gap_stops = trades[trades.get("stop_fill_type", pd.Series(index=trades.index, dtype=object)) == "gap_open"]

    summary = {
        "initial_capital": initial_capital,
        "ending_equity": ending,
        "total_return": ending / initial_capital - 1.0,
        "max_drawdown": float(daily["drawdown"].min()),
        "trades": int(len(trades)),
        "worst_trade": float(trades["net_return"].min()),
        "mean_mae": float(trades["mae"].mean()),
        "mean_mfe": float(trades["mfe"].mean()),
        "stop_trades": int(len(stop_trades)),
        "gap_stop_trades": int(len(gap_stops)),
        **risk,
    }
    return daily, summary


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
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for spec in cfg.get("candidates", []):
        trades = build_rule_trades(events, spec["rule"], int(spec["horizon"]), config["validation"])
        paths, completed = trade_paths(trades, prices, cfg.get("stop_loss"))
        equity, summary = mark_to_market_portfolio(
            paths,
            completed,
            float(cfg.get("initial_capital", 100000)),
            float(cfg.get("position_fraction", 0.05)),
        )
        name = spec["rule"] + f"_{spec['horizon']}d"
        paths.to_csv(out / f"mtm_paths_{name}.csv", index=False)
        completed.to_csv(out / f"mtm_trades_{name}.csv", index=False)
        equity.to_csv(out / f"mtm_equity_{name}.csv", index=False)
        rows.append(
            {
                "rule": spec["rule"],
                "horizon": spec["horizon"],
                **summary,
                "research_status": "gap_aware_daily_path_tested",
                "production_approved": False,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(out / "mark_to_market_summary.csv", index=False)
    print(result.to_string(index=False))
    print("\nProduction-approved candidates: 0")


if __name__ == "__main__":
    main()
