from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.atomic_io import atomic_write_csv
from src.forward_quote_capture import observation_id


ELIGIBILITY_COLUMNS = [
    "observation_id", "signal_date", "symbol", "direction", "broker_metadata_available",
    "tradable", "shortable", "easy_to_borrow", "broker_eligible", "rejection_reason",
]


def evaluate_eligibility(ledger: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=ELIGIBILITY_COLUMNS)
    metadata = snapshots.copy()
    if len(metadata):
        metadata["snapshot_date"] = pd.to_datetime(metadata["snapshot_date"])
        metadata = metadata.sort_values("snapshot_date").drop_duplicates(
            ["signal_date", "symbol", "direction"], keep="first"
        )
    lookup = {
        (str(row["signal_date"]), str(row["symbol"]), str(row["direction"])): row
        for _, row in metadata.iterrows()
    }
    rows = []
    for _, signal in ledger.iterrows():
        signal_date = str(pd.Timestamp(signal["signal_date"]).date())
        key = (signal_date, str(signal["symbol"]), str(signal["direction"]))
        asset = lookup.get(key)
        available = asset is not None
        tradable = bool(asset["tradable"]) if available and pd.notna(asset["tradable"]) else False
        shortable = bool(asset["shortable"]) if available and pd.notna(asset["shortable"]) else False
        easy = bool(asset["easy_to_borrow"]) if available and pd.notna(asset["easy_to_borrow"]) else False
        direction = str(signal["direction"])
        eligible = bool(available and tradable and (direction != "short" or (shortable and easy)))
        if not available:
            reason = "missing_broker_metadata"
        elif not tradable:
            reason = "not_tradable"
        elif direction == "short" and not shortable:
            reason = "not_shortable"
        elif direction == "short" and not easy:
            reason = "not_easy_to_borrow"
        else:
            reason = ""
        rows.append({
            "observation_id": observation_id(signal), "signal_date": signal_date, "symbol": signal["symbol"],
            "direction": direction, "broker_metadata_available": available, "tradable": tradable,
            "shortable": shortable, "easy_to_borrow": easy, "broker_eligible": eligible,
            "rejection_reason": reason,
        })
    return pd.DataFrame(rows, columns=ELIGIBILITY_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Quarantine broker-ineligible locked forward signals.")
    parser.add_argument("--ledger", default="reports/forward_observation/ledger.csv")
    parser.add_argument("--snapshots", default="reports/forward_observation/operational_snapshots.csv")
    parser.add_argument("--output", default="reports/forward_observation/eligibility.csv")
    args = parser.parse_args()

    ledger = pd.read_csv(args.ledger) if Path(args.ledger).exists() else pd.DataFrame()
    snapshots = pd.read_csv(args.snapshots) if Path(args.snapshots).exists() else pd.DataFrame()
    result = evaluate_eligibility(ledger, snapshots)
    output = Path(args.output)
    atomic_write_csv(result, output)
    eligible = int(result["broker_eligible"].sum()) if len(result) else 0
    print(f"Broker-eligible signals: {eligible}/{len(result)}. Ineligible signals are quarantined; no orders were submitted.")


if __name__ == "__main__":
    main()
