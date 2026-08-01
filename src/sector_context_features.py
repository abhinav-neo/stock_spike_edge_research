from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_sector_closes(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Sector benchmark file must use yfinance MultiIndex columns")
    field = "Adj Close" if "Adj Close" in raw.columns.get_level_values(0) else "Close"
    closes = raw[field].copy()
    closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
    closes.columns = closes.columns.astype(str)
    return closes.sort_index()


def sector_features_for_symbol(prices: pd.Series, sectors: pd.DataFrame, event_dates: pd.Series) -> pd.DataFrame:
    close = pd.to_numeric(prices, errors="coerce").sort_index()
    close.index = pd.to_datetime(close.index).normalize()
    aligned = sectors.reindex(close.index)
    stock_returns = close.pct_change()
    sector_returns = aligned.pct_change()
    prior_stock = stock_returns.shift(1)
    correlations = pd.DataFrame(index=close.index)
    for sector in aligned.columns:
        correlations[sector] = prior_stock.rolling(60, min_periods=30).corr(sector_returns[sector].shift(1))

    rows = []
    normalized_dates = pd.DatetimeIndex(pd.to_datetime(event_dates)).normalize()
    for event_date in normalized_dates:
        if event_date not in correlations.index:
            rows.append({"event_date": event_date})
            continue
        available = correlations.loc[event_date].dropna()
        if available.empty:
            rows.append({"event_date": event_date})
            continue
        sector = str(available.idxmax())
        position = aligned.index.get_loc(event_date)
        sector_close = aligned[sector]
        row = {
            "event_date": event_date,
            "inferred_sector_etf": sector,
            "prior_60d_sector_correlation": float(available.loc[sector]),
        }
        for horizon in (1, 5, 20, 60):
            row[f"sector_return_{horizon}d"] = (
                float(sector_close.iloc[position] / sector_close.iloc[position - horizon] - 1.0)
                if position >= horizon and pd.notna(sector_close.iloc[position - horizon]) and pd.notna(sector_close.iloc[position])
                else np.nan
            )
        if position >= 20 and close.iloc[position - 20] > 0 and pd.notna(row["sector_return_20d"]):
            row["stock_minus_sector_return_20d"] = float(close.iloc[position] / close.iloc[position - 20] - 1.0 - row["sector_return_20d"])
        else:
            row["stock_minus_sector_return_20d"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_sector_context(events: pd.DataFrame, sectors: pd.DataFrame, prices_dir: Path) -> pd.DataFrame:
    frames = []
    for symbol, group in events.groupby("symbol", sort=False):
        path = prices_dir / f"{symbol}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Missing price history for {symbol}")
        raw = pd.read_parquet(path)
        column = "adj_close" if "adj_close" in raw.columns else "close"
        features = sector_features_for_symbol(raw[column], sectors, group["event_date"])
        current = group.copy()
        current["event_date"] = pd.to_datetime(current["event_date"]).dt.normalize()
        frames.append(current.merge(features, on="event_date", how="left", validate="many_to_one"))
    result = pd.concat(frames, ignore_index=True)
    result["inferred_sector_etf"] = result["inferred_sector_etf"].fillna("UNKNOWN")
    return result.sort_values(["event_date", "symbol"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add leakage-safe inferred sector context to V6 events")
    parser.add_argument("--input", default="data/processed/events_features_v6.parquet")
    parser.add_argument("--sectors", default="data/raw/sector_benchmarks.parquet")
    parser.add_argument("--prices-dir", default="data/raw/prices")
    parser.add_argument("--output", default="data/processed/events_features_v6_sector.parquet")
    args = parser.parse_args()
    events = pd.read_parquet(args.input)
    sectors = load_sector_closes(Path(args.sectors))
    enriched = add_sector_context(events, sectors, Path(args.prices_dir))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output, index=False)
    enriched.to_csv(output.with_suffix(".csv"), index=False)
    print(f"Wrote {len(enriched):,} events with inferred sector context to {output}")


if __name__ == "__main__":
    main()
