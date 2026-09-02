from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import requests

from src.alpaca_historical_data import credentials_from_environment
from src.atomic_io import atomic_write_csv


TRADING_BASE_URL = "https://paper-api.alpaca.markets"
ACCOUNT_COLUMNS = [
    "snapshot_date", "status", "currency", "cash", "equity", "buying_power", "multiplier",
    "shorting_enabled", "pattern_day_trader", "daytrade_count", "trading_blocked", "account_blocked",
    "trade_suspended_by_user",
]


class AlpacaAccountClient:
    def __init__(self, key: str, secret: str, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    def account(self) -> dict:
        response = self.session.get(f"{TRADING_BASE_URL}/v2/account", timeout=30)
        response.raise_for_status()
        return response.json()


def normalize_account(payload: dict, snapshot_date: date) -> pd.DataFrame:
    row = {"snapshot_date": str(snapshot_date)}
    for column in ACCOUNT_COLUMNS[1:]:
        row[column] = payload.get(column)
    return pd.DataFrame([row], columns=ACCOUNT_COLUMNS)


def append_account_snapshots(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in (existing, new) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=ACCOUNT_COLUMNS)
    return (
        pd.concat(frames, ignore_index=True, sort=False)
        .drop_duplicates(["snapshot_date"], keep="first")
        .sort_values("snapshot_date")
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture read-only Alpaca paper-account risk controls.")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--output", default="reports/forward_observation/account_snapshots.csv")
    args = parser.parse_args()

    key, secret = credentials_from_environment()
    payload = AlpacaAccountClient(key, secret).account()
    output = Path(args.output)
    existing = pd.read_csv(output) if output.exists() and output.stat().st_size else pd.DataFrame()
    combined = append_account_snapshots(existing, normalize_account(payload, date.fromisoformat(args.date)))
    atomic_write_csv(combined, output)
    print(f"Account control snapshots: {len(combined)}. Account identifiers were not stored; no orders were submitted.")


if __name__ == "__main__":
    main()
