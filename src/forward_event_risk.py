from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.alpaca_historical_data import credentials_from_environment
from src.atomic_io import atomic_write_csv
from src.event_risk_coverage import AlpacaCorporateActionsClient, NasdaqHaltClient, annotate_candidates
from src.forward_quote_capture import observation_id


EVENT_RISK_COLUMNS = [
    "observation_id",
    "signal_date",
    "symbol",
    "captured_at_utc",
    "captured_before_entry",
    "event_day_halt",
    "reverse_split_within_30d",
    "reverse_split_within_90d",
    "reverse_split_within_180d",
    "reverse_split_within_365d",
]


def build_evidence(
    ledger: pd.DataFrame,
    actions: pd.DataFrame,
    halts: pd.DataFrame,
    captured_at: datetime,
) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=EVENT_RISK_COLUMNS)
    candidates = ledger[["signal_date", "symbol"]].copy()
    candidates["event_date"] = pd.to_datetime(candidates["signal_date"])
    candidates["period"] = "forward"
    annotated = annotate_candidates(candidates, actions, halts)
    capture = pd.Timestamp(captured_at)
    if capture.tzinfo is None:
        capture = capture.tz_localize("UTC")
    else:
        capture = capture.tz_convert("UTC")
    rows = []
    for (_, signal), (_, risk) in zip(ledger.iterrows(), annotated.iterrows()):
        entry = pd.Timestamp(signal["entry_date"])
        if entry.tzinfo is None:
            entry = entry.tz_localize("America/New_York")
        rows.append(
            {
                "observation_id": observation_id(signal),
                "signal_date": str(pd.Timestamp(signal["signal_date"]).date()),
                "symbol": signal["symbol"],
                "captured_at_utc": capture.isoformat(),
                "captured_before_entry": bool(capture < entry.tz_convert("UTC")),
                "event_day_halt": bool(risk["event_day_halt"]),
                "reverse_split_within_30d": bool(risk["reverse_split_within_30d"]),
                "reverse_split_within_90d": bool(risk["reverse_split_within_90d"]),
                "reverse_split_within_180d": bool(risk["reverse_split_within_180d"]),
                "reverse_split_within_365d": bool(risk["reverse_split_within_365d"]),
            }
        )
    return pd.DataFrame(rows, columns=EVENT_RISK_COLUMNS)


def append_first_capture(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([existing, new], ignore_index=True)
    if combined.empty:
        return pd.DataFrame(columns=EVENT_RISK_COLUMNS)
    return combined.drop_duplicates("observation_id", keep="first")[EVENT_RISK_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prospectively timestamp read-only event-risk evidence")
    parser.add_argument("--ledger", default="reports/forward_observation/ledger.csv")
    parser.add_argument("--output", default="reports/forward_observation/event_risk_evidence.csv")
    args = parser.parse_args()

    ledger = pd.read_csv(args.ledger) if Path(args.ledger).exists() else pd.DataFrame()
    output = Path(args.output)
    existing = pd.read_csv(output) if output.exists() else pd.DataFrame(columns=EVENT_RISK_COLUMNS)
    missing = ledger.loc[~ledger.apply(observation_id, axis=1).isin(set(existing.get("observation_id", [])))].copy()
    if missing.empty:
        print(f"Event-risk evidence already captured for all {len(ledger)} observations; no orders were submitted.")
        return

    signal_dates = pd.to_datetime(missing["signal_date"])
    key, secret = credentials_from_environment()
    actions = AlpacaCorporateActionsClient(key, secret).actions(
        missing["symbol"].astype(str).tolist(),
        str((signal_dates.min() - pd.Timedelta(days=365)).date()),
        str(signal_dates.max().date()),
    )
    halt_client = NasdaqHaltClient()
    halt_frames = [halt_client.halts(date) for date in sorted(signal_dates.dt.normalize().unique())]
    halts = pd.concat(halt_frames, ignore_index=True) if halt_frames else pd.DataFrame()
    evidence = build_evidence(missing, actions, halts, datetime.now(timezone.utc))
    result = append_first_capture(existing, evidence)
    atomic_write_csv(result, output)
    causal = int(evidence["captured_before_entry"].sum())
    print(f"Captured event-risk evidence for {len(evidence)} observations ({causal} before entry); no orders were submitted.")


if __name__ == "__main__":
    main()
