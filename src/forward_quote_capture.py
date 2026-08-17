from __future__ import annotations

import argparse
import hashlib
from datetime import time
from pathlib import Path

import pandas as pd

from src.alpaca_historical_data import AlpacaHistoricalClient, coverage_metrics, credentials_from_environment
from src.atomic_io import atomic_write_csv


def observation_id(row: pd.Series) -> str:
    payload = "|".join(str(row[column]) for column in ["signal_date", "candidate_key", "symbol", "direction"])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def quote_window(session_date: pd.Timestamp, phase: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    day = pd.Timestamp(session_date).date()
    if phase == "entry":
        start_local = pd.Timestamp.combine(day, time(9, 30)).tz_localize("America/New_York")
        return start_local.tz_convert("UTC"), (start_local + pd.Timedelta(minutes=5)).tz_convert("UTC")
    if phase == "exit":
        start_local = pd.Timestamp.combine(day, time(15, 55)).tz_localize("America/New_York")
        return start_local.tz_convert("UTC"), (start_local + pd.Timedelta(minutes=5)).tz_convert("UTC")
    raise ValueError(f"Unknown quote phase: {phase}")


def capture_available_windows(
    ledger: pd.DataFrame,
    client: AlpacaHistoricalClient,
    output: Path,
    feed: str = "iex",
    eligible_ids: set[str] | None = None,
) -> pd.DataFrame:
    summaries = []
    output.mkdir(parents=True, exist_ok=True)
    for _, row in ledger.iterrows():
        identifier = observation_id(row)
        if eligible_ids is not None and identifier not in eligible_ids:
            continue
        for phase, date_column in (("entry", "entry_date"), ("exit", "exit_date")):
            if date_column not in row or pd.isna(row[date_column]):
                continue
            target = output / f"observation={identifier}" / f"phase={phase}" / "quotes.parquet"
            if target.exists():
                quotes = pd.read_parquet(target)
                status = "reused"
            else:
                start, end = quote_window(pd.Timestamp(row[date_column]), phase)
                quotes = client.quotes(str(row["symbol"]), start, end, feed=feed)
                if quotes.empty:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                quotes.to_parquet(target, index=False)
                status = "downloaded"
            summaries.append({
                "observation_id": identifier,
                "signal_date": str(pd.Timestamp(row["signal_date"]).date()),
                "symbol": row["symbol"],
                "direction": row["direction"],
                "phase": phase,
                "status": status,
                **coverage_metrics(quotes),
            })
    return pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture read-only entry and exit quote windows for forward observations.")
    parser.add_argument("--ledger", default="reports/forward_observation/ledger.csv")
    parser.add_argument("--output", default="data/raw/forward_quotes")
    parser.add_argument("--summary", default="reports/forward_observation/quote_coverage.csv")
    parser.add_argument("--feed", choices=["iex", "sip"], default="iex")
    parser.add_argument("--eligibility", default="reports/forward_observation/eligibility.csv")
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    ledger = pd.read_csv(ledger_path) if ledger_path.exists() and ledger_path.stat().st_size else pd.DataFrame()
    eligibility_path = Path(args.eligibility)
    eligibility = (
        pd.read_csv(eligibility_path) if eligibility_path.exists() and eligibility_path.stat().st_size else pd.DataFrame()
    )
    eligible_ids = set(
        eligibility.loc[eligibility["broker_eligible"].fillna(False).astype(bool), "observation_id"].astype(str)
    ) if len(eligibility) else set()
    if ledger.empty:
        summary = pd.DataFrame()
    else:
        key, secret = credentials_from_environment()
        summary = capture_available_windows(
            ledger, AlpacaHistoricalClient(key, secret), Path(args.output), feed=args.feed, eligible_ids=eligible_ids
        )
    summary_path = Path(args.summary)
    atomic_write_csv(summary, summary_path)
    print(f"Captured quote windows: {len(summary)}. No orders were submitted.")


if __name__ == "__main__":
    main()
