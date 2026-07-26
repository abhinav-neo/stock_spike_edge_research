\
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def choose_price(frame: pd.DataFrame, adjusted: bool) -> pd.Series:
    if adjusted and "adj_close" in frame and frame["adj_close"].notna().any():
        return frame["adj_close"].astype(float)
    return frame["close"].astype(float)


def consecutive_days_above(values: np.ndarray, threshold: float) -> int:
    count = 0
    for value in values:
        if np.isfinite(value) and value >= threshold:
            count += 1
        else:
            break
    return count


def first_breach(values: np.ndarray, threshold: float) -> float:
    breaches = np.where(values < threshold)[0]
    return float(breaches[0] + 1) if len(breaches) else np.nan


def extract_events(path: Path, cfg: dict) -> list[dict]:
    symbol = path.stem
    df = pd.read_parquet(path).sort_index()
    if len(df) < cfg["minimum_history_days"] + max(cfg["horizons"]):
        return []

    price = choose_price(df, cfg["use_adjusted_prices"])
    prior = price.shift(1)
    daily_return = price / prior - 1
    dollar_volume = df["volume"].astype(float) * price
    adv20 = dollar_volume.shift(1).rolling(20, min_periods=10).mean()
    split = df["stock_splits"] if "stock_splits" in df else pd.Series(0.0, index=df.index)

    candidates = (
        (daily_return >= cfg["event_return_threshold"])
        & (daily_return <= cfg["maximum_event_return"])
        & (prior >= cfg["minimum_previous_close"])
        & (adv20 >= cfg["minimum_prior_20d_avg_dollar_volume"])
        & (dollar_volume >= cfg["minimum_event_day_dollar_volume"])
        & (split.fillna(0).eq(0))
    )

    rows: list[dict] = []
    positions = np.flatnonzero(candidates.to_numpy())
    for pos in positions:
        if pos + max(cfg["horizons"]) >= len(df):
            continue

        event_close = float(price.iloc[pos])
        event_open = float(df["open"].iloc[pos])
        event_high = float(df["high"].iloc[pos])
        event_low = float(df["low"].iloc[pos])
        future = price.iloc[pos + 1 : pos + max(cfg["horizons"]) + 1].to_numpy(float)
        day_range = event_high - event_low
        close_location = (event_close - event_low) / day_range if day_range > 0 else np.nan

        row = {
            "symbol": symbol,
            "event_date": df.index[pos],
            "previous_close": float(prior.iloc[pos]),
            "event_open": event_open,
            "event_high": event_high,
            "event_low": event_low,
            "event_close": event_close,
            "event_return": float(daily_return.iloc[pos]),
            "opening_gap": event_open / float(prior.iloc[pos]) - 1,
            "intraday_return": event_close / event_open - 1,
            "close_location": close_location,
            "event_volume": float(df["volume"].iloc[pos]),
            "event_dollar_volume": float(dollar_volume.iloc[pos]),
            "prior_20d_avg_dollar_volume": float(adv20.iloc[pos]),
            "relative_dollar_volume": float(dollar_volume.iloc[pos] / adv20.iloc[pos]),
            "max_forward_60d_return": float(np.nanmax(future / event_close - 1)),
            "max_forward_60d_drawdown": float(np.nanmin(future / event_close - 1)),
        }

        original_gain = event_close - float(prior.iloc[pos])
        for h in cfg["horizons"]:
            forward_close = float(price.iloc[pos + h])
            row[f"forward_return_{h}d"] = forward_close / event_close - 1
            row[f"above_event_close_{h}d"] = int(forward_close >= event_close)
            if original_gain > 0:
                retained = (forward_close - float(prior.iloc[pos])) / original_gain
                row[f"gain_retention_{h}d"] = retained

        for level in cfg["retention_levels"]:
            label = str(int(level * 100))
            row[f"consecutive_days_above_{label}pct_event_close"] = consecutive_days_above(
                future, event_close * level
            )

        for drawdown in [0.05, 0.10, 0.20, 0.30, 0.40]:
            label = str(int(drawdown * 100))
            row[f"days_to_{label}pct_breach"] = first_breach(
                future, event_close * (1 - drawdown)
            )

        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--prices", default="data/raw/prices")
    parser.add_argument("--output", default="data/processed/events.parquet")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["research"]
    events: list[dict] = []
    paths = sorted(Path(args.prices).glob("*.parquet"))
    for path in tqdm(paths, desc="Detecting events"):
        try:
            events.extend(extract_events(path, cfg))
        except Exception as exc:
            print(f"\nSkipped {path.name}: {exc}")

    result = pd.DataFrame(events)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    result.to_csv(output.with_suffix(".csv"), index=False)
    print(f"Wrote {len(result):,} events to {output}")


if __name__ == "__main__":
    main()
