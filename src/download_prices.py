\
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf
import yaml
from tqdm import tqdm


def normalize_download(raw: pd.DataFrame, requested: list[str]) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    if raw.empty:
        return result

    if isinstance(raw.columns, pd.MultiIndex):
        # yfinance may return either (Price, Ticker) or (Ticker, Price).
        level0 = set(raw.columns.get_level_values(0))
        price_names = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        ticker_first = not bool(level0 & price_names)
        for symbol in requested:
            try:
                frame = raw[symbol].copy() if ticker_first else raw.xs(symbol, axis=1, level=1).copy()
            except KeyError:
                continue
            frame = frame.dropna(how="all")
            if not frame.empty:
                result[symbol] = frame
    elif len(requested) == 1:
        result[requested[0]] = raw.dropna(how="all").copy()
    return result


def download_prices(
    symbols: list[str],
    start: str,
    end: str | None,
    output_dir: Path,
    batch_size: int,
    pause_seconds: float,
    threads: bool,
) -> tuple[list[str], list[str]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    completed, failed = [], []

    for i in tqdm(range(0, len(symbols), batch_size), desc="Downloading batches"):
        batch = symbols[i : i + batch_size]
        needed = [s for s in batch if not (output_dir / f"{s}.parquet").exists()]
        if not needed:
            completed.extend(batch)
            continue

        try:
            raw = yf.download(
                needed,
                start=start,
                end=end,
                auto_adjust=False,
                actions=True,
                group_by="column",
                threads=threads,
                progress=False,
                timeout=30,
            )
            normalized = normalize_download(raw, needed)
            for symbol in needed:
                frame = normalized.get(symbol)
                if frame is None or frame.empty:
                    failed.append(symbol)
                    continue
                frame.index = pd.to_datetime(frame.index).tz_localize(None)
                frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
                frame["symbol"] = symbol
                frame.to_parquet(output_dir / f"{symbol}.parquet", index=True)
                completed.append(symbol)
        except Exception as exc:
            print(f"\nBatch failed: {needed[:3]}...: {exc}")
            failed.extend(needed)

        time.sleep(pause_seconds)

    return completed, failed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--universe", default="data/raw/universe.csv")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    universe = pd.read_csv(args.universe)
    symbols = universe["yf_symbol"].dropna().astype(str).drop_duplicates().tolist()
    if args.limit:
        symbols = symbols[: args.limit]

    completed, failed = download_prices(
        symbols=symbols,
        start=cfg["research"]["start_date"],
        end=cfg["research"]["end_date"],
        output_dir=Path("data/raw/prices"),
        batch_size=cfg["download"]["batch_size"],
        pause_seconds=cfg["download"]["pause_seconds"],
        threads=cfg["download"]["threads"],
    )
    Path("data/raw/download_status.json").write_text(
        json.dumps({"completed": completed, "failed": failed}, indent=2)
    )
    print(f"Completed: {len(completed):,}; failed: {len(failed):,}")


if __name__ == "__main__":
    main()
