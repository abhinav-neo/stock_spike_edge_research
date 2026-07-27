from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

PRICE_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]


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
    frame = frame.rename(columns={"datetime": "date", "adj_close": "adj_close"})
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
    return (merged[PRICE_COLUMNS]
            .drop_duplicates(["symbol", "date"], keep="last")
            .sort_values(["symbol", "date"])
            .reset_index(drop=True))


def discover_symbols(existing: pd.DataFrame, orders_path: Path, explicit: list[str]) -> list[str]:
    symbols = {value.upper() for value in explicit if value.strip()}
    if not existing.empty and "symbol" in existing.columns:
        symbols.update(existing["symbol"].dropna().astype(str).str.upper())
    if orders_path.exists() and orders_path.stat().st_size:
        try:
            orders = pd.read_csv(orders_path)
            if "symbol" in orders.columns:
                symbols.update(orders["symbol"].dropna().astype(str).str.upper())
        except pd.errors.EmptyDataError:
            pass
    return sorted(symbols)


def update_market_data(output_path: Path, orders_path: Path, explicit_symbols: list[str], end_date: date) -> dict:
    existing = pd.read_parquet(output_path) if output_path.exists() else pd.DataFrame(columns=PRICE_COLUMNS)
    symbols = discover_symbols(existing, orders_path, explicit_symbols)
    if not symbols:
        raise ValueError("No symbols found. Provide --symbols or create paper orders first.")

    downloaded: list[pd.DataFrame] = []
    failures: list[str] = []
    for symbol in symbols:
        symbol_rows = existing[existing["symbol"].astype(str).str.upper().eq(symbol)] if not existing.empty else pd.DataFrame()
        start = (pd.to_datetime(symbol_rows["date"]).max().date() + timedelta(days=1)) if not symbol_rows.empty else end_date - timedelta(days=10)
        if start > end_date:
            continue
        try:
            raw = yf.download(symbol, start=start.isoformat(), end=(end_date + timedelta(days=1)).isoformat(),
                              auto_adjust=False, progress=False, threads=False)
            normalized = normalize_download(raw, symbol)
            if not normalized.empty:
                downloaded.append(normalized)
        except Exception:
            failures.append(symbol)

    updates = pd.concat(downloaded, ignore_index=True) if downloaded else pd.DataFrame(columns=PRICE_COLUMNS)
    merged = merge_prices(existing, updates)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(output_path, index=False)
    return {"symbols": len(symbols), "new_rows": len(updates), "total_rows": len(merged), "failed_symbols": failures}


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally update daily OHLCV data used by paper trading.")
    parser.add_argument("--output", default="data/processed/daily_prices.parquet")
    parser.add_argument("--orders", default="reports/paper/paper_order_blotter.csv")
    parser.add_argument("--symbols", nargs="*", default=[])
    parser.add_argument("--end-date", default=date.today().isoformat())
    args = parser.parse_args()

    summary = update_market_data(Path(args.output), Path(args.orders), args.symbols, date.fromisoformat(args.end_date))
    print(pd.DataFrame([{**summary, "failed_symbols": ",".join(summary["failed_symbols"]) }]).to_string(index=False))


if __name__ == "__main__":
    main()
