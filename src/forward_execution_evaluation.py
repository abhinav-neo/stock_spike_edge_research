from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.atomic_io import atomic_write_csv
from src.forward_quote_capture import observation_id
from src.intraday_execution import validate_quotes


EXECUTION_COLUMNS = [
    "observation_id", "signal_date", "symbol", "direction", "entry_timestamp", "entry_touch_price",
    "entry_spread_bps", "exit_timestamp", "exit_touch_price", "exit_spread_bps", "quote_gross_return",
    "quote_net_return", "bar_net_return", "execution_delta",
]


def touch_fill(quotes: pd.DataFrame, direction: str, phase: str) -> tuple[pd.Timestamp, float, float]:
    clean = validate_quotes(quotes)
    if clean.empty:
        raise ValueError("No valid quotes available for touch fill")
    quote = clean.iloc[0] if phase == "entry" else clean.iloc[-1]
    midpoint = (float(quote["bid"]) + float(quote["ask"])) / 2.0
    spread_bps = (float(quote["ask"]) - float(quote["bid"])) / midpoint * 10_000.0
    if (direction == "long" and phase == "entry") or (direction == "short" and phase == "exit"):
        price = float(quote["ask"])
    else:
        price = float(quote["bid"])
    return pd.Timestamp(quote["timestamp"]), price, spread_bps


def evaluate_executions(
    ledger: pd.DataFrame,
    quote_root: Path,
    cost_bps: float,
    minimum_entry_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    rows = []
    settled = ledger.loc[ledger["observation_status"].eq("SETTLED")] if len(ledger) else ledger
    for _, observation in settled.iterrows():
        entry_date = pd.to_datetime(observation.get("entry_date"), errors="coerce")
        if minimum_entry_date is not None and (
            pd.isna(entry_date) or entry_date < pd.Timestamp(minimum_entry_date)
        ):
            continue
        identifier = observation_id(observation)
        base = quote_root / f"observation={identifier}"
        entry_path = base / "phase=entry" / "quotes.parquet"
        exit_path = base / "phase=exit" / "quotes.parquet"
        if not entry_path.exists() or not exit_path.exists():
            continue
        direction = str(observation["direction"])
        entry_time, entry_price, entry_spread = touch_fill(pd.read_parquet(entry_path), direction, "entry")
        exit_time, exit_price, exit_spread = touch_fill(pd.read_parquet(exit_path), direction, "exit")
        gross = exit_price / entry_price - 1.0
        if direction == "short":
            gross = -gross
        net = gross - float(cost_bps) / 10_000.0
        bar_net = float(observation["net_return"])
        rows.append({
            "observation_id": identifier,
            "signal_date": str(pd.Timestamp(observation["signal_date"]).date()),
            "symbol": observation["symbol"],
            "direction": direction,
            "entry_timestamp": entry_time,
            "entry_touch_price": entry_price,
            "entry_spread_bps": entry_spread,
            "exit_timestamp": exit_time,
            "exit_touch_price": exit_price,
            "exit_spread_bps": exit_spread,
            "quote_gross_return": gross,
            "quote_net_return": net,
            "bar_net_return": bar_net,
            "execution_delta": net - bar_net,
        })
    return pd.DataFrame(rows, columns=EXECUTION_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconcile locked forward outcomes to executable quote-side fills.")
    parser.add_argument("--ledger", default="reports/forward_observation/ledger.csv")
    parser.add_argument("--quotes", default="data/raw/forward_quotes")
    parser.add_argument("--output", default="reports/forward_observation/execution_evaluation.csv")
    parser.add_argument("--cost-bps", type=float, default=100.0)
    parser.add_argument("--minimum-entry-date")
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    ledger = pd.read_csv(ledger_path) if ledger_path.exists() and ledger_path.stat().st_size else pd.DataFrame()
    result = evaluate_executions(
        ledger, Path(args.quotes), args.cost_bps,
        pd.Timestamp(args.minimum_entry_date) if args.minimum_entry_date else None,
    )
    output = Path(args.output)
    atomic_write_csv(result, output)
    print(f"Executable forward outcomes: {len(result)}. No orders were submitted.")


if __name__ == "__main__":
    main()
