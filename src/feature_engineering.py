from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


BASE_EVENT_COLUMNS = {
    "symbol",
    "event_date",
    "entry_date",
    "event_return",
    "opening_gap",
    "intraday_return",
    "close_location",
    "event_dollar_volume",
    "prior_20d_avg_dollar_volume",
    "relative_dollar_volume",
}


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan)


def build_daily_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build features using only data available through each row's close.

    All rolling reference values are shifted where necessary so event-day
    measurements are compared with information available before the event.
    """
    df = frame.sort_index().copy()
    required = {"open", "high", "low", "close", "volume"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Price frame missing required columns: {sorted(missing)}")

    close = df["adj_close"].astype(float) if "adj_close" in df else df["close"].astype(float)
    raw_close = df["close"].astype(float)
    adjustment = _safe_divide(close, raw_close)
    open_ = df["open"].astype(float) * adjustment
    high = df["high"].astype(float) * adjustment
    low = df["low"].astype(float) * adjustment
    volume = df["volume"].astype(float)
    dollar_volume = close * volume

    out = pd.DataFrame(index=df.index)
    out.index.name = "event_date"

    daily_return = close.pct_change()
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)

    for lookback in (2, 3, 5, 10, 20, 60):
        out[f"return_{lookback}d"] = close.pct_change(lookback)

    for window in (5, 10, 20, 50, 200):
        moving_average = close.rolling(window, min_periods=window).mean()
        out[f"distance_sma_{window}"] = _safe_divide(close, moving_average) - 1

    for window in (5, 10, 20, 60):
        prior_mean_volume = volume.shift(1).rolling(window, min_periods=max(3, window // 2)).mean()
        prior_mean_dollar_volume = dollar_volume.shift(1).rolling(
            window, min_periods=max(3, window // 2)
        ).mean()
        out[f"relative_volume_{window}"] = _safe_divide(volume, prior_mean_volume)
        out[f"relative_dollar_volume_{window}"] = _safe_divide(
            dollar_volume, prior_mean_dollar_volume
        )

    out["volume_acceleration_5_20"] = _safe_divide(
        volume.shift(1).rolling(5, min_periods=3).mean(),
        volume.shift(1).rolling(20, min_periods=10).mean(),
    )
    out["dollar_volume_log"] = np.log1p(dollar_volume.clip(lower=0))

    out["atr_14_pct"] = _safe_divide(
        true_range.rolling(14, min_periods=10).mean(), close.shift(1)
    )
    out["realized_volatility_10"] = daily_return.shift(1).rolling(10, min_periods=7).std()
    out["realized_volatility_20"] = daily_return.shift(1).rolling(20, min_periods=14).std()

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = _safe_divide(gain, loss)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    prior_20d_high = high.shift(1).rolling(20, min_periods=10).max()
    prior_60d_high = high.shift(1).rolling(60, min_periods=30).max()
    out["distance_prior_20d_high"] = _safe_divide(close, prior_20d_high) - 1
    out["distance_prior_60d_high"] = _safe_divide(close, prior_60d_high) - 1

    out["prior_day_return"] = daily_return.shift(1)
    out["prior_3d_return"] = close.shift(1).pct_change(3)
    out["prior_5d_return"] = close.shift(1).pct_change(5)
    out["prior_20d_return"] = close.shift(1).pct_change(20)
    out["event_open_to_prior_close"] = _safe_divide(open_, close.shift(1)) - 1
    out["event_range_pct"] = _safe_divide(high - low, close.shift(1))

    return out.replace([np.inf, -np.inf], np.nan)


def enrich_events(events: pd.DataFrame, prices_dir: Path) -> pd.DataFrame:
    missing = BASE_EVENT_COLUMNS.difference(events.columns)
    if missing:
        raise ValueError(f"Events file missing required columns: {sorted(missing)}")

    result_frames: list[pd.DataFrame] = []
    events = events.copy()
    events["event_date"] = pd.to_datetime(events["event_date"])

    for symbol, symbol_events in tqdm(events.groupby("symbol"), desc="Engineering features"):
        price_path = prices_dir / f"{symbol}.parquet"
        if not price_path.exists():
            continue

        prices = pd.read_parquet(price_path)
        if not isinstance(prices.index, pd.DatetimeIndex):
            if "date" not in prices.columns:
                continue
            prices = prices.set_index(pd.to_datetime(prices["date"]))

        daily_features = build_daily_features(prices)
        current = symbol_events.merge(
            daily_features.reset_index(),
            how="left",
            on="event_date",
            validate="many_to_one",
        )
        result_frames.append(current)

    if not result_frames:
        return events.iloc[0:0].copy()

    enriched = pd.concat(result_frames, ignore_index=True)
    enriched = enriched.sort_values(["event_date", "symbol"]).reset_index(drop=True)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V5 leakage-safe event features")
    parser.add_argument("--events", default="data/processed/events.parquet")
    parser.add_argument("--prices", default="data/raw/prices")
    parser.add_argument("--output", default="data/processed/events_features_v5.parquet")
    args = parser.parse_args()

    events_path = Path(args.events)
    if not events_path.exists():
        raise FileNotFoundError(f"Events file not found: {events_path}")

    events = pd.read_parquet(events_path)
    enriched = enrich_events(events, Path(args.prices))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(output, index=False)
    enriched.to_csv(output.with_suffix(".csv"), index=False)

    engineered_count = len(set(enriched.columns).difference(events.columns))
    print(f"Wrote {len(enriched):,} events with {engineered_count} engineered features to {output}")


if __name__ == "__main__":
    main()
