from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


QUOTE_COLUMNS = {"symbol", "timestamp", "bid", "ask", "bid_size", "ask_size"}
SIGNAL_COLUMNS = {"symbol", "signal_timestamp", "exit_timestamp", "side", "quantity"}


@dataclass(frozen=True)
class ExecutionCosts:
    latency_ms: int = 100
    commission_per_share: float = 0.0
    minimum_commission: float = 0.0
    regulatory_fee_bps_on_sales: float = 0.0
    impact_bps_at_full_touch: float = 2.0
    maximum_participation: float = 0.25


def validate_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    missing = QUOTE_COLUMNS - set(quotes.columns)
    if missing:
        raise ValueError(f"quotes missing required columns: {sorted(missing)}")
    result = quotes.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    numeric = ["bid", "ask", "bid_size", "ask_size"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    if result[numeric].isna().any().any():
        raise ValueError("quotes contain non-numeric or missing prices/sizes")
    if (result[["bid", "ask"]] <= 0).any().any() or (result[["bid_size", "ask_size"]] < 0).any().any():
        raise ValueError("quotes contain non-positive prices or negative sizes")
    if (result["ask"] < result["bid"]).any():
        raise ValueError("quotes contain crossed markets")
    if result.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("quotes contain duplicate symbol/timestamp rows")
    result = result.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return result


def validate_signals(signals: pd.DataFrame) -> pd.DataFrame:
    missing = SIGNAL_COLUMNS - set(signals.columns)
    if missing:
        raise ValueError(f"signals missing required columns: {sorted(missing)}")
    result = signals.copy()
    for column in ("signal_timestamp", "exit_timestamp"):
        result[column] = pd.to_datetime(result[column], utc=True)
    if (~result["side"].isin(["buy", "sell"])).any():
        raise ValueError("signal side must be buy or sell")
    result["quantity"] = pd.to_numeric(result["quantity"], errors="coerce")
    if result["quantity"].isna().any() or (result["quantity"] <= 0).any():
        raise ValueError("signal quantity must be positive")
    if (result["exit_timestamp"] <= result["signal_timestamp"]).any():
        raise ValueError("exit_timestamp must follow signal_timestamp")
    return result.sort_values("signal_timestamp").reset_index(drop=True)


def _fill(quote: pd.Series, side: str, quantity: float, costs: ExecutionCosts) -> tuple[float, float]:
    touch_price = float(quote["ask"] if side == "buy" else quote["bid"])
    touch_size = float(quote["ask_size"] if side == "buy" else quote["bid_size"])
    permitted = touch_size * costs.maximum_participation
    if touch_size <= 0 or quantity > permitted:
        return np.nan, 0.0
    participation = quantity / touch_size
    impact_bps = costs.impact_bps_at_full_touch * np.sqrt(participation)
    direction = 1.0 if side == "buy" else -1.0
    return touch_price * (1.0 + direction * impact_bps / 10_000.0), quantity


def _first_quote_after(quotes: pd.DataFrame, timestamp: pd.Timestamp) -> pd.Series | None:
    positions = quotes["timestamp"].searchsorted(timestamp, side="left")
    if positions >= len(quotes):
        return None
    return quotes.iloc[int(positions)]


def simulate_round_trips(
    quotes: pd.DataFrame,
    signals: pd.DataFrame,
    costs: ExecutionCosts = ExecutionCosts(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Execute marketable round trips without using quotes before order arrival."""
    quote_data = validate_quotes(quotes)
    signal_data = validate_signals(signals)
    grouped = {symbol: frame.reset_index(drop=True) for symbol, frame in quote_data.groupby("symbol")}
    fills: list[dict] = []
    rejections: list[dict] = []
    latency = pd.Timedelta(milliseconds=costs.latency_ms)
    for trade_id, signal in signal_data.iterrows():
        symbol_quotes = grouped.get(signal["symbol"])
        if symbol_quotes is None:
            rejections.append({"trade_id": trade_id, "reason": "no_symbol_quotes"})
            continue
        entry_quote = _first_quote_after(symbol_quotes, signal["signal_timestamp"] + latency)
        exit_quote = _first_quote_after(symbol_quotes, signal["exit_timestamp"] + latency)
        if entry_quote is None or exit_quote is None:
            rejections.append({"trade_id": trade_id, "reason": "no_quote_after_arrival"})
            continue
        quantity = float(signal["quantity"])
        exit_side = "sell" if signal["side"] == "buy" else "buy"
        entry_price, entry_quantity = _fill(entry_quote, signal["side"], quantity, costs)
        exit_price, exit_quantity = _fill(exit_quote, exit_side, quantity, costs)
        if entry_quantity < quantity or exit_quantity < quantity:
            rejections.append({"trade_id": trade_id, "reason": "touch_participation_limit"})
            continue
        direction = 1.0 if signal["side"] == "buy" else -1.0
        gross_pnl = direction * (exit_price - entry_price) * quantity
        commission = 2.0 * max(costs.minimum_commission, costs.commission_per_share * quantity)
        sale_notional = (exit_price if signal["side"] == "buy" else entry_price) * quantity
        regulatory_fee = sale_notional * costs.regulatory_fee_bps_on_sales / 10_000.0
        net_pnl = gross_pnl - commission - regulatory_fee
        fills.append(
            {
                "trade_id": trade_id,
                "symbol": signal["symbol"],
                "side": signal["side"],
                "quantity": quantity,
                "entry_timestamp": entry_quote["timestamp"],
                "exit_timestamp": exit_quote["timestamp"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_pnl": gross_pnl,
                "commission": commission,
                "regulatory_fee": regulatory_fee,
                "net_pnl": net_pnl,
                "gross_return_on_notional": gross_pnl / (entry_price * quantity),
                "net_return_on_notional": net_pnl / (entry_price * quantity),
            }
        )
    return pd.DataFrame(fills), pd.DataFrame(rejections)


def account_metrics(fills: pd.DataFrame, initial_capital: float = 10_000.0) -> tuple[pd.DataFrame, dict]:
    if fills.empty:
        return pd.DataFrame(), {"initial_capital": initial_capital, "ending_equity": initial_capital, "trades": 0}
    ordered = fills.sort_values("exit_timestamp").copy()
    ordered["equity"] = initial_capital + ordered["net_pnl"].cumsum()
    ordered["drawdown"] = ordered["equity"] / ordered["equity"].cummax().clip(lower=initial_capital) - 1.0
    elapsed_days = max((ordered["exit_timestamp"].iloc[-1] - ordered["entry_timestamp"].iloc[0]).total_seconds() / 86_400, 1.0)
    years = elapsed_days / 365.25
    ending = float(ordered["equity"].iloc[-1])
    cagr = (ending / initial_capital) ** (1 / years) - 1 if ending > 0 else np.nan
    trading_days = max(ordered["exit_timestamp"].dt.normalize().nunique(), 1)
    summary = {
        "initial_capital": initial_capital,
        "ending_equity": ending,
        "total_return": ending / initial_capital - 1,
        "cagr": cagr,
        "max_drawdown": float(ordered["drawdown"].min()),
        "trades": int(len(ordered)),
        "trades_per_day": float(len(ordered) / trading_days),
        "win_rate": float((ordered["net_pnl"] > 0).mean()),
        "gross_pnl": float(ordered["gross_pnl"].sum()),
        "net_pnl": float(ordered["net_pnl"].sum()),
        "explicit_costs": float((ordered["commission"] + ordered["regulatory_fee"]).sum()),
    }
    return ordered, summary
