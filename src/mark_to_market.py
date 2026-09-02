from __future__ import annotations

import argparse
import hashlib
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


def apply_locate_model(
    trades: pd.DataFrame,
    availability_probability: float = 1.0,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a deterministic pseudo-random historical locate-availability model.

    This does not claim to reconstruct actual historical locates. It is a scenario
    test that reproducibly rejects a configured fraction of otherwise valid trades.
    """
    probability = float(availability_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("availability_probability must be between 0 and 1")
    if trades.empty or probability >= 1.0:
        return trades.copy(), trades.iloc[0:0].copy()

    accepted = []
    for _, row in trades.iterrows():
        event_date = pd.Timestamp(row["event_date"]).strftime("%Y-%m-%d")
        key = f"{random_seed}|{row['symbol']}|{event_date}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / float(2**64)
        accepted.append(value < probability)
    mask = pd.Series(accepted, index=trades.index)
    return trades.loc[mask].copy(), trades.loc[~mask].copy()


def margin_liquidation_price_multiple(initial_margin: float, maintenance_margin: float) -> float:
    """Return the short-price multiple where position equity reaches maintenance margin.

    The model assumes short-sale proceeds plus the initial margin deposit remain as
    collateral. It is a position-level stress rule, not a substitute for a broker's
    account-wide house-margin calculation.
    """
    initial = float(initial_margin)
    maintenance = float(maintenance_margin)
    if initial < 0:
        raise ValueError("initial margin must be non-negative")
    if maintenance < 0:
        raise ValueError("maintenance margin must be non-negative")
    return (1.0 + initial) / (1.0 + maintenance)


def trade_paths(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    stop_loss: float | None = None,
    short_borrow_bps_annual: float = 0.0,
    initial_margin_requirement: float | None = None,
    maintenance_margin_requirement: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct daily paths for short trades with gap-aware stops and borrow cost."""
    validate_prices(prices)
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["symbol", "date"])
    daily_borrow_rate = float(short_borrow_bps_annual) / 10000.0 / 252.0
    margin_multiple = None
    if initial_margin_requirement is not None or maintenance_margin_requirement is not None:
        if initial_margin_requirement is None or maintenance_margin_requirement is None:
            raise ValueError("initial and maintenance margin requirements must be supplied together")
        margin_multiple = margin_liquidation_price_multiple(
            initial_margin_requirement, maintenance_margin_requirement
        )

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
        holding_days = len(after)
        mae = 0.0
        mfe = 0.0
        stop_price = entry * (1.0 + float(stop_loss)) if stop_loss is not None else None
        margin_price = entry * margin_multiple if margin_multiple is not None else None

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
            liquidated = False
            mark_price = day_close
            trigger_prices = [price for price in (stop_price, margin_price) if price is not None]
            trigger_price = min(trigger_prices) if trigger_prices else None
            if trigger_price is not None:
                if day_open >= trigger_price:
                    stopped = True
                    mark_price = day_open
                    stop_fill_type = "gap_open"
                elif day_high >= trigger_price:
                    stopped = True
                    mark_price = trigger_price
                    stop_fill_type = "intraday_stop"
                liquidated = bool(
                    stopped and margin_price is not None and margin_price <= (stop_price or float("inf"))
                )

            gross_mark_return = (entry - mark_price) / entry
            accrued_borrow_return = daily_borrow_rate * day_number
            mark_return = gross_mark_return - accrued_borrow_return
            path_rows.append(
                {
                    "trade_id": trade_id,
                    "symbol": symbol,
                    "date": row["date"],
                    "day": day_number,
                    "gross_mark_return": gross_mark_return,
                    "accrued_borrow_return": accrued_borrow_return,
                    "mark_return": mark_return,
                }
            )

            if stopped:
                exit_reason = "margin_liquidation" if liquidated else "stop"
                exit_price = mark_price
                exit_date = row["date"]
                holding_days = day_number
                break

        gross_return = (entry - exit_price) / entry
        borrow_cost_return = daily_borrow_rate * holding_days
        net_return = gross_return - borrow_cost_return
        trade_rows.append(
            {
                "trade_id": trade_id,
                "symbol": symbol,
                "entry_date": after.iloc[0]["date"],
                "exit_date": exit_date,
                "entry_price": entry,
                "exit_price": exit_price,
                "holding_days": holding_days,
                "gross_return": gross_return,
                "borrow_cost_return": borrow_cost_return,
                "net_return": net_return,
                "mae": mae,
                "mfe": mfe,
                "exit_reason": exit_reason,
                "stop_fill_type": stop_fill_type,
                "margin_liquidation_price_multiple": margin_multiple,
            }
        )

    return pd.DataFrame(path_rows), pd.DataFrame(trade_rows)


def _risk_statistics(equity: pd.DataFrame) -> dict:
    if equity.empty or len(equity) < 2:
        return {"cagr": 0.0, "annualized_volatility": 0.0, "sharpe": np.nan, "sortino": np.nan, "calmar": np.nan, "active_day_sharpe": np.nan}

    daily_returns = equity["equity"].pct_change().replace([np.inf, -np.inf], np.nan)
    valid_returns = daily_returns.dropna()
    elapsed_days = max((equity["date"].iloc[-1] - equity["date"].iloc[0]).days, 1)
    years = elapsed_days / 365.25
    start_equity = float(equity["equity"].iloc[0])
    end_equity = float(equity["equity"].iloc[-1])
    cagr = (end_equity / start_equity) ** (1.0 / years) - 1.0 if start_equity > 0 and end_equity > 0 else np.nan

    volatility = float(valid_returns.std(ddof=1) * np.sqrt(252)) if len(valid_returns) > 1 else 0.0
    mean_daily = float(valid_returns.mean()) if len(valid_returns) else 0.0
    daily_std = float(valid_returns.std(ddof=1)) if len(valid_returns) > 1 else 0.0
    downside = valid_returns[valid_returns < 0]
    downside_std = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sharpe = mean_daily / daily_std * np.sqrt(252) if daily_std > 0 else np.nan
    sortino = mean_daily / downside_std * np.sqrt(252) if downside_std > 0 else np.nan
    max_drawdown = abs(float(equity["drawdown"].min()))
    calmar = cagr / max_drawdown if max_drawdown > 0 and np.isfinite(cagr) else np.nan

    active_returns = daily_returns[equity["concurrent_positions"].gt(0)].dropna()
    active_std = float(active_returns.std(ddof=1)) if len(active_returns) > 1 else 0.0
    active_day_sharpe = float(active_returns.mean()) / active_std * np.sqrt(252) if active_std > 0 else np.nan
    return {
        "cagr": float(cagr),
        "annualized_volatility": volatility,
        "sharpe": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "sortino": float(sortino) if np.isfinite(sortino) else np.nan,
        "calmar": float(calmar) if np.isfinite(calmar) else np.nan,
        "active_day_sharpe": float(active_day_sharpe) if np.isfinite(active_day_sharpe) else np.nan,
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
            "active_days_pct": 0.0,
            "average_gross_exposure": 0.0,
            "maximum_gross_exposure": 0.0,
            "average_concurrent_positions": 0.0,
            "maximum_concurrent_positions": 0,
        }

    stake = initial_capital * position_fraction
    paths = paths.copy()
    trades = trades.copy()
    paths["date"] = pd.to_datetime(paths["date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    trades["pnl"] = trades["net_return"] * stake

    dates = pd.Index(sorted(paths["date"].unique()), name="date")
    current_marks = paths.assign(mark_pnl=paths["mark_return"] * stake).groupby("date")["mark_pnl"].sum()
    concurrent = paths.groupby("date")["trade_id"].nunique().reindex(dates, fill_value=0).astype(int)
    realized_by_date = trades.groupby("exit_date")["pnl"].sum().reindex(dates, fill_value=0.0)
    realized_before_date = realized_by_date.cumsum().shift(1, fill_value=0.0)

    daily = pd.DataFrame(index=dates)
    daily["realized_pnl_prior"] = realized_before_date
    daily["active_mark_pnl"] = current_marks.reindex(dates, fill_value=0.0)
    daily["concurrent_positions"] = concurrent
    daily["gross_exposure"] = daily["concurrent_positions"] * position_fraction
    daily["equity"] = initial_capital + daily["realized_pnl_prior"] + daily["active_mark_pnl"]
    daily["drawdown"] = daily["equity"] / daily["equity"].cummax() - 1.0
    daily = daily.reset_index()

    ending = initial_capital + float(trades["pnl"].sum())
    risk = _risk_statistics(daily)
    stop_trades = trades[trades["exit_reason"] == "stop"] if "exit_reason" in trades else trades.iloc[0:0]
    gap_stops = trades[trades.get("stop_fill_type", pd.Series(index=trades.index, dtype=object)) == "gap_open"]
    avg_exposure = float(daily["gross_exposure"].mean())
    annualized_return_on_deployed = risk["cagr"] / avg_exposure if avg_exposure > 0 and np.isfinite(risk["cagr"]) else np.nan

    summary = {
        "initial_capital": initial_capital,
        "ending_equity": ending,
        "total_return": ending / initial_capital - 1.0,
        "max_drawdown": float(daily["drawdown"].min()),
        "trades": int(len(trades)),
        "worst_trade": float(trades["net_return"].min()),
        "mean_mae": float(trades["mae"].mean()),
        "mean_mfe": float(trades["mfe"].mean()),
        "total_borrow_cost": float((trades["borrow_cost_return"] * stake).sum()) if "borrow_cost_return" in trades else 0.0,
        "stop_trades": int(len(stop_trades)),
        "gap_stop_trades": int(len(gap_stops)),
        "active_days_pct": float(daily["concurrent_positions"].gt(0).mean()),
        "average_gross_exposure": avg_exposure,
        "maximum_gross_exposure": float(daily["gross_exposure"].max()),
        "average_concurrent_positions": float(daily["concurrent_positions"].mean()),
        "maximum_concurrent_positions": int(daily["concurrent_positions"].max()),
        "annualized_return_on_deployed_capital": float(annualized_return_on_deployed) if np.isfinite(annualized_return_on_deployed) else np.nan,
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
        candidate_trades = build_rule_trades(events, spec["rule"], int(spec["horizon"]), config["validation"])
        trades, rejected_locates = apply_locate_model(
            candidate_trades,
            float(cfg.get("locate_availability_probability", 1.0)),
            int(cfg.get("locate_random_seed", 42)),
        )
        paths, completed = trade_paths(
            trades,
            prices,
            cfg.get("stop_loss"),
            float(cfg.get("short_borrow_bps_annual", 0.0)),
        )
        equity, summary = mark_to_market_portfolio(
            paths,
            completed,
            float(cfg.get("initial_capital", 100000)),
            float(cfg.get("position_fraction", 0.05)),
        )
        name = spec["rule"] + f"_{spec['horizon']}d"
        paths.to_csv(out / f"mtm_paths_{name}.csv", index=False)
        completed.to_csv(out / f"mtm_trades_{name}.csv", index=False)
        rejected_locates.to_csv(out / f"mtm_rejected_locates_{name}.csv", index=False)
        equity.to_csv(out / f"mtm_equity_{name}.csv", index=False)
        rows.append(
            {
                "rule": spec["rule"],
                "horizon": spec["horizon"],
                "candidate_trades": int(len(candidate_trades)),
                "locate_rejections": int(len(rejected_locates)),
                "locate_availability_probability": float(cfg.get("locate_availability_probability", 1.0)),
                "short_borrow_bps_annual": float(cfg.get("short_borrow_bps_annual", 0.0)),
                **summary,
                "research_status": "borrow_and_utilization_tested",
                "production_approved": False,
            }
        )

    result = pd.DataFrame(rows)
    result.to_csv(out / "mark_to_market_summary.csv", index=False)
    print(result.to_string(index=False))
    print("\nProduction-approved candidates: 0")


if __name__ == "__main__":
    main()
