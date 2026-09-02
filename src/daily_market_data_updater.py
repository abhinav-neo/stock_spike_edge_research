from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
from itertools import islice
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from src.atomic_io import atomic_write_parquet

PRICE_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]


def latest_weekday(value: date) -> date:
    """Return the latest possible U.S. trading date without a calendar dependency."""
    while value.weekday() >= 5:
        value -= timedelta(days=1)
    return value


def latest_completed_session(value: date, now: datetime | None = None) -> date:
    """Avoid requesting a daily bar before the provider can reasonably publish it."""
    eastern_now = now or datetime.now(ZoneInfo("America/New_York"))
    if eastern_now.tzinfo is None:
        eastern_now = eastern_now.replace(tzinfo=ZoneInfo("America/New_York"))
    else:
        eastern_now = eastern_now.astimezone(ZoneInfo("America/New_York"))
    candidate = min(value, eastern_now.date())
    if candidate == eastern_now.date() and eastern_now.time() < time(18, 0):
        candidate -= timedelta(days=1)
    return latest_weekday(candidate)


def normalize_download(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    frame = raw.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        if symbol in frame.columns.get_level_values(-1):
            frame = frame.xs(symbol, axis=1, level=-1)
        else:
            frame.columns = frame.columns.get_level_values(0)
    frame = frame.reset_index()
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    frame = frame.rename(columns={"datetime": "date"})
    if "date" not in frame.columns:
        raise ValueError(f"Downloaded data for {symbol} has no date column")
    frame["symbol"] = symbol.upper()
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    if "adj_close" not in frame.columns and "close" in frame.columns:
        frame["adj_close"] = frame["close"]
    for column in PRICE_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[PRICE_COLUMNS].dropna(subset=["date", "open", "high", "low", "close"])


def merge_prices(existing: pd.DataFrame, updates: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in (existing, updates) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=PRICE_COLUMNS)
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged["symbol"] = merged["symbol"].astype(str).str.upper()
    merged["date"] = pd.to_datetime(merged["date"]).dt.tz_localize(None).dt.normalize()
    for column in PRICE_COLUMNS:
        if column not in merged.columns:
            merged[column] = pd.NA
    ordered_columns = PRICE_COLUMNS + [column for column in merged.columns if column not in PRICE_COLUMNS]
    return (merged[ordered_columns]
            .drop_duplicates(["symbol", "date"], keep="last")
            .sort_values(["symbol", "date"])
            .reset_index(drop=True))


def discover_symbols(
    existing: pd.DataFrame, orders_path: Path, explicit: list[str], full_existing_universe: bool = False
) -> list[str]:
    symbols = {value.upper() for value in explicit if value.strip()}
    if orders_path.exists() and orders_path.stat().st_size:
        try:
            orders = pd.read_csv(orders_path)
            if "symbol" in orders.columns:
                symbols.update(orders["symbol"].dropna().astype(str).str.upper())
        except pd.errors.EmptyDataError:
            pass
    if (full_existing_universe or not symbols) and not existing.empty and "symbol" in existing.columns:
        symbols.update(existing["symbol"].dropna().astype(str).str.upper())
    return sorted(symbols)


def batches(values: list[str], size: int) -> list[list[str]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    iterator = iter(values)
    result = []
    while chunk := list(islice(iterator, size)):
        result.append(chunk)
    return result


def download_batch(
    symbols: list[str], starts: dict[str, date], end_date: date, timeout: float = 5.0
) -> tuple[list[pd.DataFrame], list[str]]:
    start = min(starts[symbol] for symbol in symbols)
    tickers: str | list[str] = symbols[0] if len(symbols) == 1 else symbols
    raw = yf.download(
        tickers, start=start.isoformat(), end=(end_date + timedelta(days=1)).isoformat(),
        auto_adjust=False, progress=False, threads=len(symbols) > 1, group_by="column",
        timeout=timeout,
    )
    downloaded: list[pd.DataFrame] = []
    no_data: list[str] = []
    for symbol in symbols:
        normalized = normalize_download(raw, symbol)
        normalized = normalized.loc[normalized["date"].dt.date.ge(starts[symbol])]
        if normalized.empty:
            no_data.append(symbol)
        else:
            downloaded.append(normalized)
    return downloaded, no_data


def update_market_data(
    output_path: Path,
    orders_path: Path,
    explicit_symbols: list[str],
    end_date: date,
    batch_size: int = 100,
    full_existing_universe: bool = False,
    download_timeout: float = 5.0,
) -> dict:
    if download_timeout <= 0:
        raise ValueError("download timeout must be positive")
    requested_end_date = end_date
    end_date = latest_completed_session(end_date)
    existing = pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame(columns=PRICE_COLUMNS)
    symbols = discover_symbols(existing, orders_path, explicit_symbols, full_existing_universe)
    if not symbols:
        raise ValueError("No symbols found. Provide --symbols or create paper orders first.")

    downloaded: list[pd.DataFrame] = []
    failures: list[str] = []
    no_data: list[str] = []
    starts: dict[str, date] = {}
    latest_dates: dict[str, date] = {}
    if not existing.empty:
        dated = existing[["symbol", "date"]].copy()
        dated["symbol"] = dated["symbol"].astype(str).str.upper()
        dated["date"] = pd.to_datetime(dated["date"], errors="coerce")
        maxima = dated.dropna(subset=["date"]).groupby("symbol", sort=False)["date"].max()
        latest_dates = {symbol: timestamp.date() for symbol, timestamp in maxima.items()}
    for symbol in symbols:
        last_date = latest_dates.get(symbol)
        start = last_date + timedelta(days=1) if last_date else end_date - timedelta(days=10)
        if start <= end_date:
            starts[symbol] = start

    for chunk in batches(sorted(starts), batch_size):
        try:
            frames, empty = download_batch(chunk, starts, end_date, download_timeout)
            downloaded.extend(frames)
            no_data.extend(empty)
        except Exception:
            for symbol in chunk:
                try:
                    frames, empty = download_batch([symbol], starts, end_date, download_timeout)
                    downloaded.extend(frames)
                    no_data.extend(empty)
                except Exception:
                    failures.append(symbol)

    updates = pd.concat(downloaded, ignore_index=True) if downloaded else pd.DataFrame(columns=PRICE_COLUMNS)
    merged = merge_prices(existing, updates)
    atomic_write_parquet(merged, output_path)
    return {
        "symbols": len(symbols), "new_rows": len(updates), "total_rows": len(merged),
        "requested_end_date": str(requested_end_date), "effective_end_date": str(end_date),
        "failed_symbols": failures, "no_data_symbols": no_data,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally update daily OHLCV data used by paper trading.")
    parser.add_argument("--output", default="data/processed/daily_prices.parquet")
    parser.add_argument("--orders", default="reports/paper/paper_order_blotter.csv")
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--download-timeout", type=float, default=5.0)
    parser.add_argument("--full-existing-universe", action="store_true")
    args = parser.parse_args()

    summary = update_market_data(
        Path(args.output), Path(args.orders), args.symbols, date.fromisoformat(args.end_date), args.batch_size,
        args.full_existing_universe, args.download_timeout,
    )
    printable = {
        **summary,
        "failed_symbols": ",".join(summary["failed_symbols"]),
        "no_data_symbols": ",".join(summary["no_data_symbols"]),
    }
    print(pd.DataFrame([printable]).to_string(index=False))


if __name__ == "__main__":
    main()
