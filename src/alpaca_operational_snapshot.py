from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from src.alpaca_historical_data import credentials_from_environment
from src.atomic_io import atomic_write_csv


TRADING_BASE_URL = "https://paper-api.alpaca.markets"
SNAPSHOT_COLUMNS = [
    "snapshot_date", "signal_date", "symbol", "direction", "asset_id", "asset_status", "tradable",
    "shortable", "easy_to_borrow", "fractionable", "maintenance_margin_requirement",
    "borrow_status",
]


class AlpacaAssetClient:
    def __init__(self, key: str, secret: str, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    def asset(self, symbol: str) -> dict:
        response = self.session.get(f"{TRADING_BASE_URL}/v2/assets/{symbol}", timeout=30)
        response.raise_for_status()
        return response.json()


def collect_snapshots(ledger: pd.DataFrame, client: AlpacaAssetClient, snapshot_date: date) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    signals = ledger.loc[pd.to_datetime(ledger["signal_date"]).dt.date.le(snapshot_date)].copy()
    rows = []
    for _, signal in signals.drop_duplicates(["signal_date", "symbol", "direction"]).iterrows():
        asset = client.asset(str(signal["symbol"]))
        rows.append({
            "snapshot_date": str(snapshot_date),
            "signal_date": str(pd.Timestamp(signal["signal_date"]).date()),
            "symbol": signal["symbol"],
            "direction": signal["direction"],
            "asset_id": asset.get("id"),
            "asset_status": asset.get("status"),
            "tradable": asset.get("tradable"),
            "shortable": asset.get("shortable"),
            "easy_to_borrow": asset.get("easy_to_borrow"),
            "fractionable": asset.get("fractionable"),
            "maintenance_margin_requirement": asset.get("maintenance_margin_requirement"),
            "borrow_status": asset.get("borrow_status"),
        })
    return pd.DataFrame(rows, columns=SNAPSHOT_COLUMNS)


def append_snapshots(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in (existing, new) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)
    keys = ["snapshot_date", "signal_date", "symbol", "direction"]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    for column in SNAPSHOT_COLUMNS:
        if column not in combined:
            combined[column] = pd.NA
    # Preserve the first captured value while allowing a later schema version to
    # backfill newly introduced fields such as borrow_status on the same date.
    combined = combined.groupby(keys, as_index=False, sort=False).agg(
        {column: lambda values: values.dropna().iloc[0] if values.notna().any() else pd.NA
         for column in SNAPSHOT_COLUMNS if column not in keys}
    )
    return combined[SNAPSHOT_COLUMNS].sort_values(keys).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture read-only Alpaca asset metadata for locked forward signals.")
    parser.add_argument("--ledger", default="reports/forward_observation/ledger.csv")
    parser.add_argument("--output", default="reports/forward_observation/operational_snapshots.csv")
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()

    ledger_path = Path(args.ledger)
    ledger = pd.read_csv(ledger_path) if ledger_path.exists() and ledger_path.stat().st_size else pd.DataFrame()
    output = Path(args.output)
    existing = pd.read_csv(output) if output.exists() and output.stat().st_size else pd.DataFrame()
    if ledger.empty:
        combined = append_snapshots(existing, pd.DataFrame())
    else:
        key, secret = credentials_from_environment()
        combined = append_snapshots(
            existing, collect_snapshots(ledger, AlpacaAssetClient(key, secret), date.fromisoformat(args.date))
        )
    atomic_write_csv(combined, output)
    print(f"Operational snapshots: {len(combined)}. No orders were submitted.")


if __name__ == "__main__":
    main()
