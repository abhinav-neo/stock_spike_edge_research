from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from src.intraday_execution import validate_quotes


DATA_BASE_URL = "https://data.alpaca.markets"


def credentials_from_environment() -> tuple[str, str]:
    key = os.environ.get("APCA_API_KEY_ID", "").strip()
    secret = os.environ.get("APCA_API_SECRET_KEY", "").strip()
    if not key or not secret:
        raise RuntimeError(
            "Missing Alpaca credentials. Set APCA_API_KEY_ID and APCA_API_SECRET_KEY locally; do not commit them."
        )
    return key, secret


class AlpacaHistoricalClient:
    def __init__(
        self,
        key: str,
        secret: str,
        session: requests.Session | None = None,
        base_url: str = DATA_BASE_URL,
        retries: int = 5,
    ) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        self.base_url = base_url.rstrip("/")
        self.retries = retries

    def _get(self, path: str, params: dict) -> dict:
        for attempt in range(self.retries):
            response = self.session.get(f"{self.base_url}{path}", params=params, timeout=60)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 == self.retries:
                    response.raise_for_status()
                retry_after = float(response.headers.get("Retry-After", 2**attempt))
                time.sleep(min(retry_after, 30.0))
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("Alpaca request exhausted retries")

    def quotes(
        self,
        symbol: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        feed: str = "iex",
        limit: int = 10_000,
    ) -> pd.DataFrame:
        rows: list[dict] = []
        token = None
        while True:
            params = {
                "start": pd.Timestamp(start).isoformat(),
                "end": pd.Timestamp(end).isoformat(),
                "feed": feed,
                "limit": limit,
                "sort": "asc",
            }
            if token:
                params["page_token"] = token
            payload = self._get(f"/v2/stocks/{symbol}/quotes", params)
            for quote in payload.get("quotes", []):
                rows.append(
                    {
                        "symbol": symbol,
                        "timestamp": quote["t"],
                        "bid": quote["bp"],
                        "ask": quote["ap"],
                        "bid_size": quote["bs"],
                        "ask_size": quote["as"],
                        "bid_exchange": quote.get("bx"),
                        "ask_exchange": quote.get("ax"),
                        "conditions": quote.get("c"),
                        "tape": quote.get("z"),
                    }
                )
            token = payload.get("next_page_token")
            if not token:
                break
        if not rows:
            return pd.DataFrame(columns=["symbol", "timestamp", "bid", "ask", "bid_size", "ask_size"])
        frame = pd.DataFrame(rows)
        validated = validate_quotes(frame[["symbol", "timestamp", "bid", "ask", "bid_size", "ask_size"]])
        extras = frame.drop(columns=["bid", "ask", "bid_size", "ask_size"]).copy()
        extras["timestamp"] = pd.to_datetime(extras["timestamp"], utc=True)
        extras = extras.drop_duplicates(["symbol", "timestamp"], keep="last")
        return validated.merge(extras, on=["symbol", "timestamp"], how="left", validate="one_to_one")


def regular_session_quotes(quotes: pd.DataFrame) -> pd.DataFrame:
    if quotes.empty:
        return quotes.copy()
    local = quotes["timestamp"].dt.tz_convert("America/New_York")
    minutes = local.dt.hour * 60 + local.dt.minute
    return quotes.loc[(minutes >= 9 * 60 + 30) & (minutes < 16 * 60)].copy()


def coverage_metrics(quotes: pd.DataFrame) -> dict:
    if quotes.empty:
        return {"quotes": 0, "symbols": 0, "trading_days": 0}
    local = quotes["timestamp"].dt.tz_convert("America/New_York")
    midpoint = (quotes["bid"] + quotes["ask"]) / 2
    spread_bps = (quotes["ask"] - quotes["bid"]) / midpoint * 10_000
    return {
        "quotes": int(len(quotes)),
        "symbols": int(quotes["symbol"].nunique()),
        "trading_days": int(local.dt.date.nunique()),
        "start": str(quotes["timestamp"].min()),
        "end": str(quotes["timestamp"].max()),
        "median_spread_bps": float(spread_bps.median()),
        "p95_spread_bps": float(spread_bps.quantile(0.95)),
        "crossed_quotes": int((quotes["bid"] > quotes["ask"]).sum()),
        "zero_size_quotes": int(((quotes["bid_size"] <= 0) | (quotes["ask_size"] <= 0)).sum()),
    }


def download_partitions(
    client: AlpacaHistoricalClient,
    symbols: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    output: Path,
    feed: str = "iex",
) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for symbol in symbols:
        for day in pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D", inclusive="left"):
            next_day = day + pd.Timedelta(days=1)
            quotes = regular_session_quotes(client.quotes(symbol, day, next_day, feed=feed))
            if quotes.empty:
                continue
            target = output / f"symbol={symbol}" / f"date={day.date()}" / "quotes.parquet"
            target.parent.mkdir(parents=True, exist_ok=True)
            quotes.to_parquet(target, index=False)
            summaries.append({"symbol": symbol, "date": str(day.date()), **coverage_metrics(quotes)})
    return pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download partitioned Alpaca historical stock quotes")
    parser.add_argument("--symbols", nargs="+", default=["SPY", "QQQ", "AAPL", "MSFT", "NVDA"])
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--feed", choices=["iex", "sip"], default="iex")
    parser.add_argument("--output", default="data/raw/alpaca_quotes")
    parser.add_argument("--summary", default="reports/alpaca_data_coverage.csv")
    args = parser.parse_args()
    key, secret = credentials_from_environment()
    client = AlpacaHistoricalClient(key, secret)
    summary = download_partitions(
        client,
        args.symbols,
        pd.Timestamp(args.start, tz="UTC"),
        pd.Timestamp(args.end, tz="UTC"),
        Path(args.output),
        args.feed,
    )
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
