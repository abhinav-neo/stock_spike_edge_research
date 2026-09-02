\
from __future__ import annotations

import io
import os
import re
from pathlib import Path
import pandas as pd
import requests


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
ALPHA_VANTAGE_URL = "https://www.alphavantage.co/query"


def _read_pipe_file(url: str) -> pd.DataFrame:
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    text = response.text
    rows = [line for line in text.splitlines() if not line.startswith("File Creation Time")]
    return pd.read_csv(io.StringIO("\n".join(rows)), sep="|")


def load_current_us_universe(
    exclude_etfs: bool = True,
    exclude_test_issues: bool = True,
    exclude_special_securities: bool = True,
) -> pd.DataFrame:
    """Build a current U.S.-listed universe from Nasdaq Trader symbol directories."""
    nasdaq = _read_pipe_file(NASDAQ_LISTED_URL)
    nasdaq = nasdaq.rename(
        columns={
            "Symbol": "symbol",
            "Security Name": "name",
            "ETF": "is_etf",
            "Test Issue": "test_issue",
        }
    )
    nasdaq["exchange"] = "NASDAQ"

    other = _read_pipe_file(OTHER_LISTED_URL)
    other = other.rename(
        columns={
            "ACT Symbol": "symbol",
            "Security Name": "name",
            "ETF": "is_etf",
            "Test Issue": "test_issue",
            "Exchange": "exchange_code",
        }
    )
    exchange_map = {
        "A": "NYSE American",
        "N": "NYSE",
        "P": "NYSE Arca",
        "Z": "Cboe",
        "V": "IEX",
    }
    other["exchange"] = other["exchange_code"].map(exchange_map).fillna(other["exchange_code"])

    keep = ["symbol", "name", "exchange", "is_etf", "test_issue"]
    universe = pd.concat([nasdaq[keep], other[keep]], ignore_index=True)
    universe = universe.dropna(subset=["symbol"]).drop_duplicates("symbol")

    if exclude_etfs:
        universe = universe[universe["is_etf"].astype(str).str.upper().ne("Y")]
    if exclude_test_issues:
        universe = universe[universe["test_issue"].astype(str).str.upper().ne("Y")]

    # Remove obvious warrants, rights, units, preferreds and other non-common issues.
    if exclude_special_securities:
        bad_name = re.compile(
            r"\b(WARRANT|RIGHT|RIGHTS|UNIT|UNITS|PREFERRED|DEPOSITARY|"
            r"NOTE|NOTES|BOND|ETF|ETN|FUND|BENEFICIAL INTEREST)\b",
            re.IGNORECASE,
        )
        universe = universe[~universe["name"].fillna("").str.contains(bad_name)]
        universe = universe[~universe["symbol"].str.contains(r"[\^\./$]", regex=True)]

    # Yahoo uses '-' instead of '.' for many share classes.
    universe["yf_symbol"] = universe["symbol"].str.replace(".", "-", regex=False)
    return universe.sort_values("symbol").reset_index(drop=True)


def download_alpha_vantage_listing_status(
    api_key: str,
    state: str = "active",
    date: str | None = None,
) -> pd.DataFrame:
    """Download active/delisted metadata. Free key required."""
    if state not in {"active", "delisted"}:
        raise ValueError("state must be 'active' or 'delisted'")
    params = {"function": "LISTING_STATUS", "state": state, "apikey": api_key}
    if date:
        params["date"] = date
    response = requests.get(ALPHA_VANTAGE_URL, params=params, timeout=60)
    response.raise_for_status()
    data = pd.read_csv(io.StringIO(response.text))
    if len(data.columns) == 1:
        raise RuntimeError(
            "Alpha Vantage did not return listing CSV. Check the API key or rate limit."
        )
    data["state_requested"] = state
    return data


def build_historical_metadata(api_key: str | None, output: Path) -> pd.DataFrame:
    """Combine current and delisted metadata where a free Alpha Vantage key is available."""
    current = load_current_us_universe()
    current["status"] = "Active"
    frames = [current]

    if api_key:
        try:
            delisted = download_alpha_vantage_listing_status(api_key, state="delisted")
            delisted = delisted.rename(
                columns={
                    "symbol": "symbol",
                    "name": "name",
                    "exchange": "exchange",
                    "delistingDate": "delisting_date",
                    "ipoDate": "ipo_date",
                    "assetType": "asset_type",
                    "status": "status",
                }
            )
            delisted["yf_symbol"] = delisted["symbol"].str.replace(".", "-", regex=False)
            frames.append(delisted)
        except Exception as exc:
            print(f"Warning: delisted metadata unavailable: {exc}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates(["symbol", "status"], keep="first")
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    return combined


if __name__ == "__main__":
    key = os.getenv("ALPHA_VANTAGE_API_KEY")
    out = Path("data/raw/universe.csv")
    frame = build_historical_metadata(key, out)
    print(f"Wrote {len(frame):,} universe records to {out}")
