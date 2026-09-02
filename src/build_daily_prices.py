from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

from src.event_study import prepare_price_frame


OUTPUT_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]


def normalize_daily_frame(path: Path, use_adjusted_prices: bool) -> pd.DataFrame:
    """Load one raw symbol file and return normalized daily OHLCV rows."""
    raw = pd.read_parquet(path).sort_index()
    prepared = prepare_price_frame(raw, use_adjusted_prices)

    missing = [column for column in ["open", "high", "low", "close", "volume"] if column not in prepared]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    date_values = pd.to_datetime(prepared.index).tz_localize(None)
    symbol = (
        prepared["symbol"].astype(str)
        if "symbol" in prepared
        else pd.Series(path.stem, index=prepared.index, dtype="object")
    )

    result = pd.DataFrame(
        {
            "symbol": symbol.to_numpy(),
            "date": date_values,
            "open": pd.to_numeric(prepared["open"], errors="coerce").to_numpy(),
            "high": pd.to_numeric(prepared["high"], errors="coerce").to_numpy(),
            "low": pd.to_numeric(prepared["low"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(prepared["close"], errors="coerce").to_numpy(),
            "volume": pd.to_numeric(prepared["volume"], errors="coerce").to_numpy(),
        }
    )
    result = result.dropna(subset=["symbol", "date", "open", "high", "low", "close"])
    result = result[(result[["open", "high", "low", "close"]] > 0).all(axis=1)]
    return result[OUTPUT_COLUMNS]


def build_daily_prices(
    input_dir: Path,
    output: Path,
    use_adjusted_prices: bool = True,
    limit: int | None = None,
) -> tuple[int, int, int]:
    """Stream per-symbol Parquet files into one consolidated Parquet dataset."""
    paths = sorted(input_dir.glob("*.parquet"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(
            f"No raw price files found in {input_dir}. Run python -m src.download_prices first."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)

    writer: pq.ParquetWriter | None = None
    written_rows = 0
    completed = 0
    skipped = 0
    try:
        for path in tqdm(paths, desc="Building daily price dataset"):
            try:
                frame = normalize_daily_frame(path, use_adjusted_prices)
                if frame.empty:
                    skipped += 1
                    continue
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="snappy")
                writer.write_table(table)
                written_rows += len(frame)
                completed += 1
            except Exception as exc:
                skipped += 1
                print(f"\nSkipped {path.name}: {exc}")
    finally:
        if writer is not None:
            writer.close()

    if writer is None or written_rows == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("No valid daily price rows were generated")

    output.unlink(missing_ok=True)
    temporary.replace(output)
    return written_rows, completed, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input-dir", default="data/raw/prices")
    parser.add_argument("--output", default="data/processed/daily_prices.parquet")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["research"]
    rows, completed, skipped = build_daily_prices(
        Path(args.input_dir),
        Path(args.output),
        use_adjusted_prices=bool(cfg.get("use_adjusted_prices", True)),
        limit=args.limit,
    )
    print(f"Wrote {rows:,} daily rows from {completed:,} symbols to {args.output}")
    print(f"Skipped symbols: {skipped:,}")


if __name__ == "__main__":
    main()
