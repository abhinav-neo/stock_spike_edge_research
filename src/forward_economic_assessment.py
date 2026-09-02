from __future__ import annotations

import numpy as np
import pandas as pd

from src.forward_quote_capture import observation_id


def select_capacity(
    trades: pd.DataFrame,
    max_daily_entries: int,
    max_concurrent_positions: int,
    prevent_symbol_overlap: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return trades.copy(), trades.copy()
    accepted: list[dict] = []
    rejected: list[dict] = []
    active: list[dict] = []
    daily_counts: dict[pd.Timestamp, int] = {}
    ordered = trades.sort_values(["entry_date", "candidate_rank", "symbol"])
    for _, trade in ordered.iterrows():
        entry_date = pd.Timestamp(trade["entry_date"]).normalize()
        active = [position for position in active if position["exit_date"] >= entry_date]
        active_symbols = {position["symbol"] for position in active}
        reason = None
        if daily_counts.get(entry_date, 0) >= max_daily_entries:
            reason = "daily_entry_cap"
        elif len(active) >= max_concurrent_positions:
            reason = "concurrency_cap"
        elif prevent_symbol_overlap and str(trade["symbol"]) in active_symbols:
            reason = "symbol_overlap"
        item = trade.to_dict()
        if reason:
            item["rejection_reason"] = reason
            rejected.append(item)
            continue
        accepted.append(item)
        active.append({"symbol": str(trade["symbol"]), "exit_date": pd.Timestamp(trade["exit_date"]).normalize()})
        daily_counts[entry_date] = daily_counts.get(entry_date, 0) + 1
    return pd.DataFrame(accepted), pd.DataFrame(rejected)


def executable_trades(ledger: pd.DataFrame, executions: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty or executions.empty:
        return pd.DataFrame()
    keyed = ledger.copy()
    keyed["observation_id"] = keyed.apply(observation_id, axis=1)
    return keyed.merge(executions, on=["observation_id", "signal_date", "symbol", "direction"], how="inner")


def capital_reserving_metrics(
    trades: pd.DataFrame,
    prices: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    initial_capital = float(config.get("initial_capital", 100_000.0))
    position_fraction = float(config.get("position_fraction", 0.05))
    max_daily = int(config.get("max_daily_entries", 3))
    max_concurrent = int(config.get("max_concurrent_positions", 10))
    prevent_overlap = bool(config.get("prevent_symbol_overlap", True))
    target_cagr = float(config.get("target_cagr", 0.40))
    maximum_drawdown = float(config.get("maximum_acceptable_drawdown", 0.25))
    maximum_gross_exposure = float(config.get("maximum_gross_exposure", 1.0))
    short_borrow_rate = float(config.get("short_borrow_bps_annual", 0.0)) / 10_000.0
    base = {
        "economic_initial_capital": initial_capital,
        "economic_position_fraction": position_fraction,
        "economic_target_cagr": target_cagr,
        "economic_maximum_acceptable_drawdown": maximum_drawdown,
        "economic_maximum_gross_exposure": maximum_gross_exposure,
        "economic_short_borrow_bps_annual": short_borrow_rate * 10_000.0,
        "economic_accepted_trades": 0,
        "economic_rejected_trades": 0,
        "economic_cagr": None,
        "economic_max_drawdown": None,
        "economic_mtm_coverage": 0.0,
        "economic_max_gross_exposure": None,
        "economic_minimum_equity": None,
        "economic_gate_passed": False,
    }
    if trades.empty:
        return trades.copy(), pd.DataFrame(), base

    normalized = trades.copy()
    for column in ["entry_date", "exit_date"]:
        normalized[column] = pd.to_datetime(normalized[column]).dt.normalize()
    target_notional = initial_capital * position_fraction
    affordable = pd.to_numeric(normalized["entry_touch_price"], errors="coerce").le(target_notional)
    insufficient_notional = int((~affordable).sum())
    accepted, rejected = select_capacity(
        normalized.loc[affordable], max_daily, max_concurrent, prevent_overlap
    )
    if accepted.empty:
        return accepted, pd.DataFrame(), {
            **base, "economic_rejected_trades": int(len(rejected)) + insufficient_notional,
        }

    price_data = prices[["date", "symbol", "close"]].copy()
    price_data["date"] = pd.to_datetime(price_data["date"]).dt.normalize()
    close_lookup = price_data.drop_duplicates(["date", "symbol"], keep="last").set_index(["date", "symbol"])["close"]
    dates = sorted(price_data.loc[
        price_data["date"].between(accepted["entry_date"].min(), accepted["exit_date"].max()), "date"
    ].unique())
    cash = initial_capital
    notional = target_notional
    open_positions: list[dict] = []
    curve_rows: list[dict] = []
    required_marks = available_marks = 0

    for raw_date in dates:
        day = pd.Timestamp(raw_date).normalize()
        closing = [position for position in open_positions if position["exit_date"] == day]
        open_positions = [position for position in open_positions if position["exit_date"] != day]
        for position in closing:
            direction = 1.0 if position["direction"] == "long" else -1.0
            pnl = direction * position["shares"] * (position["exit_touch_price"] - position["entry_touch_price"])
            cash += position["notional"] + pnl

        entries = accepted.loc[accepted["entry_date"].eq(day)]
        for _, trade in entries.iterrows():
            cost_rate = float(trade["quote_gross_return"]) - float(trade["quote_net_return"])
            available_notional = cash / (1.0 + cost_rate) if cost_rate > -1.0 else 0.0
            allocated = min(notional, max(available_notional, 0.0))
            shares = np.floor(allocated / float(trade["entry_touch_price"]))
            if shares < 1:
                continue
            allocated = float(shares * float(trade["entry_touch_price"]))
            cash -= allocated * (1.0 + cost_rate)
            item = trade.to_dict()
            item["notional"] = allocated
            item["shares"] = shares
            open_positions.append(item)

        marked_value = 0.0
        marked_gross = 0.0
        daily_borrow_cost = 0.0
        for position in open_positions:
            required_marks += 1
            key = (day, position["symbol"])
            if key not in close_lookup.index:
                continue
            available_marks += 1
            mark = float(close_lookup.loc[key])
            market_value = position["shares"] * mark
            marked_gross += market_value
            if position["direction"] == "short":
                daily_borrow_cost += market_value * short_borrow_rate / 365.25
            direction = 1.0 if position["direction"] == "long" else -1.0
            pnl = direction * position["shares"] * (mark - position["entry_touch_price"])
            marked_value += position["notional"] + pnl
        cash -= daily_borrow_cost
        equity = cash + marked_value
        gross_exposure = marked_gross / equity if equity > 0 else float("inf")
        curve_rows.append({
            "date": day, "cash": cash, "open_positions": len(open_positions),
            "daily_borrow_cost": daily_borrow_cost, "gross_exposure": gross_exposure,
            "equity": equity,
        })

    curve = pd.DataFrame(curve_rows)
    curve["peak"] = curve["equity"].cummax().clip(lower=initial_capital)
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1.0
    span_days = max((accepted["exit_date"].max() - accepted["entry_date"].min()).days, 1)
    ending_equity = float(curve.iloc[-1]["equity"])
    cagr = (ending_equity / initial_capital) ** (365.25 / span_days) - 1.0 if ending_equity > 0 else -1.0
    drawdown = float(-curve["drawdown"].min())
    coverage = float(available_marks / required_marks) if required_marks else 0.0
    max_observed_gross = float(curve["gross_exposure"].max())
    minimum_equity = float(curve["equity"].min())
    metrics = {
        **base,
        "economic_accepted_trades": int(len(accepted)),
        "economic_rejected_trades": int(len(rejected)) + insufficient_notional,
        "economic_ending_equity": ending_equity,
        "economic_cagr": float(cagr) if np.isfinite(cagr) else None,
        "economic_max_drawdown": drawdown,
        "economic_mtm_coverage": coverage,
        "economic_max_gross_exposure": max_observed_gross,
        "economic_minimum_equity": minimum_equity,
    }
    metrics["economic_gate_passed"] = bool(
        coverage >= 1.0
        and minimum_equity > 0
        and max_observed_gross <= maximum_gross_exposure
        and cagr >= target_cagr
        and drawdown <= maximum_drawdown
    )
    return accepted, curve, metrics
