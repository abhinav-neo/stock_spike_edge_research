from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.atomic_io import atomic_write_csv
from src.forward_locate_evidence import append_locate_evidence


def broker_etb_evidence(eligibility: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    if eligibility.empty or snapshots.empty:
        return pd.DataFrame()
    metadata = snapshots.copy()
    metadata["snapshot_date"] = pd.to_datetime(metadata["snapshot_date"])
    metadata = metadata.sort_values("snapshot_date").drop_duplicates(
        ["signal_date", "symbol", "direction"], keep="first"
    )
    metadata = metadata.rename(columns={"borrow_status": "snapshot_borrow_status", "easy_to_borrow": "snapshot_easy"})
    joined = eligibility.merge(
        metadata[["signal_date", "symbol", "direction", "snapshot_date", "snapshot_borrow_status", "snapshot_easy"]],
        on=["signal_date", "symbol", "direction"], how="left", validate="one_to_one",
    )
    etb = joined.loc[
        joined["broker_eligible"].fillna(False).astype(bool)
        & joined["direction"].eq("short")
        & (joined["snapshot_borrow_status"].eq("easy_to_borrow") | joined["snapshot_easy"].fillna(False).astype(bool))
    ].copy()
    if etb.empty:
        return pd.DataFrame()
    return pd.DataFrame({
        "observation_id": etb["observation_id"],
        "decision_timestamp": pd.to_datetime(etb["snapshot_date"], utc=True),
        "provider": "alpaca",
        "locate_requested": False,
        "locate_confirmed": True,
        "quoted_borrow_rate_annual": 0.0,
        "available_quantity": pd.NA,
        "source_reference": "alpaca_asset_borrow_status:easy_to_borrow",
        "locate_basis": "broker_etb",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Record Alpaca-established ETB locate evidence; never requests HTB locates")
    parser.add_argument("--eligibility", default="reports/forward_observation/eligibility.csv")
    parser.add_argument("--snapshots", default="reports/forward_observation/operational_snapshots.csv")
    parser.add_argument("--output", default="reports/forward_observation/locate_evidence.csv")
    args = parser.parse_args()

    eligibility = pd.read_csv(args.eligibility) if Path(args.eligibility).exists() else pd.DataFrame()
    snapshots = pd.read_csv(args.snapshots) if Path(args.snapshots).exists() else pd.DataFrame()
    output = Path(args.output)
    existing = pd.read_csv(output) if output.exists() and output.stat().st_size else pd.DataFrame()
    additions = broker_etb_evidence(eligibility, snapshots)
    combined = append_locate_evidence(existing, additions)
    atomic_write_csv(combined, output)
    print(f"Broker-established ETB locate records: {len(combined)}. HTB locate requests were not submitted.")


if __name__ == "__main__":
    main()
