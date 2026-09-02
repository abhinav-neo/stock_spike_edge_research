from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.v5_mtm_research import normalize_spy


def normalize_single_benchmark(path: Path, output_column: str) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(column[0]).lower().replace(" ", "_") for column in raw.columns]
    else:
        raw.columns = [str(column).lower().replace(" ", "_") for column in raw.columns]
    raw = raw.reset_index()
    date_column = next(column for column in raw.columns if column.lower() in {"date", "datetime"})
    value_column = "adj_close" if "adj_close" in raw.columns else "close"
    result = raw[[date_column, value_column]].rename(columns={date_column: "date", value_column: output_column})
    result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None).dt.normalize()
    return result


def build_market_features(spy: pd.DataFrame) -> pd.DataFrame:
    frame = spy.copy().sort_values("date")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    close = pd.to_numeric(frame["benchmark"], errors="coerce")
    output = pd.DataFrame({"event_date": frame["date"]})
    for horizon in (1, 5, 20, 60):
        output[f"spy_return_{horizon}d"] = close.pct_change(horizon)
    for window in (20, 50, 200):
        average = close.rolling(window, min_periods=window).mean()
        output[f"spy_distance_sma_{window}"] = close / average - 1.0
    returns = close.pct_change()
    output["spy_realized_volatility_20"] = returns.rolling(20, min_periods=15).std()
    output["spy_drawdown_252"] = close / close.rolling(252, min_periods=100).max() - 1.0
    output["spy_above_sma_200"] = (output["spy_distance_sma_200"] >= 0).astype("Int64")
    return output.replace([np.inf, -np.inf], np.nan)


def build_vix_features(vix: pd.DataFrame) -> pd.DataFrame:
    frame = vix.copy().sort_values("date")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    close = pd.to_numeric(frame["vix_close"], errors="coerce")
    output = pd.DataFrame({"event_date": frame["date"], "vix_close": close})
    output["vix_return_1d"] = close.pct_change()
    output["vix_return_5d"] = close.pct_change(5)
    output["vix_distance_sma_20"] = close / close.rolling(20, min_periods=15).mean() - 1.0
    output["vix_percentile_252"] = close.rolling(252, min_periods=100).rank(pct=True)
    return output.replace([np.inf, -np.inf], np.nan)


def add_market_context(events: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    left = events.copy()
    right = market.copy()
    left["event_date"] = pd.to_datetime(left["event_date"]).dt.normalize()
    right["event_date"] = pd.to_datetime(right["event_date"]).dt.normalize()
    if right.duplicated("event_date").any():
        raise ValueError("Market feature data contains duplicate event dates")
    result = left.merge(right, on="event_date", how="left", validate="many_to_one")
    market_columns = [column for column in right.columns if column != "event_date"]
    missing_rows = result[market_columns].isna().all(axis=1)
    if missing_rows.any():
        dates = result.loc[missing_rows, "event_date"].dt.strftime("%Y-%m-%d").unique().tolist()
        raise ValueError(f"Missing market context for event dates: {dates[:10]}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Add point-in-time SPY context to V5 event features")
    parser.add_argument("--input", default="data/processed/events_features_v5.parquet")
    parser.add_argument("--spy", default="data/raw/spy_benchmark.parquet")
    parser.add_argument("--vix", default=None)
    parser.add_argument("--output", default="data/processed/events_features_v6.parquet")
    args = parser.parse_args()

    events = pd.read_parquet(args.input)
    market = build_market_features(normalize_spy(Path(args.spy)))
    if args.vix:
        vix = build_vix_features(normalize_single_benchmark(Path(args.vix), "vix_close"))
        market = market.merge(vix, on="event_date", how="left", validate="one_to_one")
    enriched = add_market_context(events, market)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output, index=False)
    enriched.to_csv(output.with_suffix(".csv"), index=False)
    print(f"Wrote {len(enriched):,} events with {len(market.columns) - 1} market features to {output}")


if __name__ == "__main__":
    main()
