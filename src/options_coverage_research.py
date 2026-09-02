from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

from src.alpaca_historical_data import credentials_from_environment
from src.v5_mtm_research import attach_event_dates, build_period_trades


CONTRACTS_URL = "https://paper-api.alpaca.markets/v2/options/contracts"
HISTORICAL_OPTION_DATA_START = pd.Timestamp("2024-02-01")


class AlpacaOptionsClient:
    def __init__(self, key: str, secret: str, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    def puts(self, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
        params = {
            "underlying_symbols": symbol,
            "status": "inactive",
            "type": "put",
            "expiration_date_gte": str(pd.Timestamp(start).date()),
            "expiration_date_lte": str(pd.Timestamp(end).date()),
            "limit": 10000,
        }
        contracts: list[dict] = []
        while True:
            response = self.session.get(CONTRACTS_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            contracts.extend(payload.get("option_contracts") or [])
            token = payload.get("next_page_token")
            if not token:
                return contracts
            params["page_token"] = token


def coverage(candidates: pd.DataFrame, client: AlpacaOptionsClient) -> pd.DataFrame:
    rows = []
    for row in candidates.itertuples(index=False):
        event_date = pd.Timestamp(row.event_date).normalize()
        if event_date < HISTORICAL_OPTION_DATA_START:
            rows.append({
                "symbol": row.symbol, "event_date": event_date, "data_available": False,
                "put_contracts": 0, "expirations": 0, "strikes": 0, "spread_constructible": False,
            })
            continue
        contracts = client.puts(row.symbol, event_date + pd.Timedelta(days=14), event_date + pd.Timedelta(days=45))
        expirations = {item.get("expiration_date") for item in contracts if item.get("expiration_date")}
        strikes = {item.get("strike_price") for item in contracts if item.get("strike_price")}
        rows.append({
            "symbol": row.symbol, "event_date": event_date, "data_available": True,
            "put_contracts": len(contracts), "expirations": len(expirations), "strikes": len(strikes),
            "spread_constructible": bool(len(expirations) > 0 and len(strikes) >= 2),
        })
    return pd.DataFrame(rows)


def write_assessment(result: pd.DataFrame) -> str:
    available = result.loc[result["data_available"]]
    constructible = int(available["spread_constructible"].sum()) if len(available) else 0
    fraction = float(available["spread_constructible"].mean()) if len(available) else 0.0
    return f"""# Bounded-Loss Options Coverage Assessment

## Verdict

**Not eligible for historical promotion.** Alpaca historical options data begins in
February 2024. Of {len(result)} locked test candidates, {len(available)} occur inside
that window and {constructible} ({fraction:.1%}) have at least one 14–45 DTE expiration
with two put strikes, the minimum contract topology for a vertical spread.

Contract topology alone is not execution evidence. The connected data API provides
historical trades and bars but no historical bid/ask quote endpoint; the free option
feed is indicative rather than actual OPRA. Therefore entry debit, exit credit, spread,
slippage, and fill feasibility cannot be reconstructed honestly. No return, CAGR, or
drawdown is calculated for this path, and test-period contract availability is not used
to tune the stock model.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure bounded-loss put-spread contract coverage")
    parser.add_argument("--predictions", default="reports/v5_validation_suite/model/predictions.csv")
    parser.add_argument("--features", default="data/processed/events_features_v5.parquet")
    parser.add_argument("--output-dir", default="reports/options_coverage")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    features = pd.read_parquet(args.features)
    candidates = attach_event_dates(build_period_trades(predictions, features, "test"), features, 5)
    key, secret = credentials_from_environment()
    result = coverage(candidates, AlpacaOptionsClient(key, secret))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "candidate_contract_coverage.csv", index=False)
    summary = {
        "candidates": len(result),
        "inside_data_window": int(result["data_available"].sum()),
        "spread_constructible": int(result["spread_constructible"].sum()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "ASSESSMENT.md").write_text(write_assessment(result), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
