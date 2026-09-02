from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from src.intraday_execution import validate_quotes


def _level_volume(level) -> float:
    return float(sum(order.volume for order in level.queue))


def top_quote(lob, price_scale: float = 10_000.0) -> tuple[float, float, float, float] | None:
    if lob is None or not lob.bid_levels or not lob.ask_levels:
        return None
    bid, ask = lob.bid_levels[0], lob.ask_levels[0]
    return (
        float(bid.price) / price_scale,
        float(ask.price) / price_scale,
        _level_volume(bid),
        _level_volume(ask),
    )


def extract_itch2_quotes(
    source: Path,
    symbol: str,
    session_date: dt.date,
    session_start: dt.time = dt.time(9, 30),
    session_end: dt.time = dt.time(16, 0),
) -> pd.DataFrame:
    try:
        from meatpy.itch2.itch2_market_processor import ITCH2MarketProcessor
        from meatpy.itch2.itch2_message_reader import ITCH2MessageReader
    except ImportError as error:
        raise RuntimeError("meatpy is required to parse Nasdaq ITCH samples") from error

    processor = ITCH2MarketProcessor(symbol, dt.datetime.combine(session_date, dt.time()))
    rows: list[dict] = []
    pending_timestamp: int | None = None
    pending_quote: tuple[float, float, float, float] | None = None

    def emit() -> None:
        if pending_timestamp is None or pending_quote is None:
            return
        local = dt.datetime.combine(session_date, dt.time()) + dt.timedelta(milliseconds=pending_timestamp)
        if not session_start <= local.time() <= session_end:
            return
        timestamp = pd.Timestamp(local, tz="America/New_York").tz_convert("UTC")
        bid, ask, bid_size, ask_size = pending_quote
        rows.append(
            {
                "symbol": symbol,
                "timestamp": timestamp,
                "bid": bid,
                "ask": ask,
                "bid_size": bid_size,
                "ask_size": ask_size,
            }
        )

    previous_quote = None
    for message in ITCH2MessageReader().read_file(source):
        message_timestamp = int(message.timestamp)
        if pending_timestamp is not None and message_timestamp != pending_timestamp:
            emit()
            pending_quote = None
        pending_timestamp = message_timestamp
        processor.process_message(message)
        current = top_quote(processor.lob)
        if current is not None and current != previous_quote:
            pending_quote = current
            previous_quote = current
    emit()
    if not rows:
        return pd.DataFrame(columns=["symbol", "timestamp", "bid", "ask", "bid_size", "ask_size"])
    quotes = pd.DataFrame(rows).drop_duplicates(["symbol", "timestamp"], keep="last")
    return validate_quotes(quotes)


def quote_diagnostics(quotes: pd.DataFrame) -> dict:
    if quotes.empty:
        return {"quotes": 0}
    midpoint = (quotes["bid"] + quotes["ask"]) / 2
    spread_bps = (quotes["ask"] - quotes["bid"]) / midpoint * 10_000
    elapsed_hours = max((quotes["timestamp"].iloc[-1] - quotes["timestamp"].iloc[0]).total_seconds() / 3600, 1e-9)
    return {
        "quotes": int(len(quotes)),
        "start": str(quotes["timestamp"].iloc[0]),
        "end": str(quotes["timestamp"].iloc[-1]),
        "quote_changes_per_hour": float(len(quotes) / elapsed_hours),
        "median_spread_bps": float(spread_bps.median()),
        "p95_spread_bps": float(spread_bps.quantile(0.95)),
        "median_bid_size": float(quotes["bid_size"].median()),
        "median_ask_size": float(quotes["ask_size"].median()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract top-of-book quotes from the public Nasdaq ITCH 2.0 sample")
    parser.add_argument("--source", default="data/raw/S010303-v2.zip")
    parser.add_argument("--symbol", default="MSFT")
    parser.add_argument("--date", default="2003-01-03")
    parser.add_argument("--output", default="data/processed/nasdaq_itch2_msft_quotes.parquet")
    args = parser.parse_args()
    quotes = extract_itch2_quotes(Path(args.source), args.symbol, dt.date.fromisoformat(args.date))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    quotes.to_parquet(output, index=False)
    print(quote_diagnostics(quotes))


if __name__ == "__main__":
    main()
