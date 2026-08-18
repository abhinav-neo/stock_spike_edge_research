from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from src.alpaca_historical_data import credentials_from_environment
from src.free_data_factorial_research import evaluate_variant


BARS_URL = "https://data.alpaca.markets/v2/stocks/bars"


class AlpacaMinuteBarsClient:
    def __init__(self, key: str, secret: str, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    def event_day(self, symbols: list[str], date: pd.Timestamp, feed: str = "sip") -> pd.DataFrame:
        day = pd.Timestamp(date).strftime("%Y-%m-%d")
        params = {
            "symbols": ",".join(sorted(set(symbols))), "timeframe": "1Min",
            "start": f"{day}T09:30:00-04:00", "end": f"{day}T16:00:00-04:00",
            "feed": feed, "limit": 10000, "adjustment": "raw", "asof": day, "sort": "asc",
        }
        rows = []
        while True:
            response = self.session.get(BARS_URL, params=params, timeout=60)
            if response.status_code == 429:
                time.sleep(float(response.headers.get("Retry-After", 3)))
                continue
            response.raise_for_status()
            payload = response.json()
            for symbol, bars in (payload.get("bars") or {}).items():
                for bar in bars:
                    rows.append({
                        "symbol": symbol, "timestamp": bar["t"], "open": bar["o"],
                        "high": bar["h"], "low": bar["l"], "close": bar["c"],
                        "volume": bar["v"], "vwap": bar.get("vw"), "trades": bar.get("n"),
                    })
            token = payload.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)


def summarize_intraday(events: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    work = bars.copy()
    if not work.empty:
        work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True)
        work["event_date"] = work["timestamp"].dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    rows = []
    grouped = (
        {(symbol, date): group for (symbol, date), group in work.groupby(["symbol", "event_date"])}
        if not work.empty else {}
    )
    for event in events.itertuples(index=False):
        date = pd.Timestamp(event.event_date).normalize()
        group = grouped.get((str(event.symbol), date), pd.DataFrame()).copy()
        base = {"symbol": event.symbol, "event_date": date, "intraday_bar_count": len(group)}
        if len(group) < 30:
            rows.append(base)
            continue
        group = group.sort_values("timestamp")
        opening, closing = float(group.iloc[0]["open"]), float(group.iloc[-1]["close"])
        high_index = int(np.argmax(group["high"].to_numpy(float)))
        minute_returns = group["close"].astype(float).pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        total_volume = float(group["volume"].sum())
        timestamps = group["timestamp"].dt.tz_convert("America/New_York")
        minutes = timestamps.dt.hour * 60 + timestamps.dt.minute
        gaps = timestamps.diff().dt.total_seconds().div(60).dropna()
        base.update({
            "intraday_open_close_return": closing / opening - 1,
            "intraday_high_from_open": float(group["high"].max()) / opening - 1,
            "intraday_low_from_open": float(group["low"].min()) / opening - 1,
            "intraday_high_to_close_reversal": closing / float(group["high"].max()) - 1,
            "intraday_high_time_fraction": high_index / max(len(group) - 1, 1),
            "intraday_realized_volatility": float(np.sqrt(np.square(minute_returns).sum())),
            "intraday_max_1m_return": float(minute_returns.max()),
            "intraday_min_1m_return": float(minute_returns.min()),
            "intraday_first30_return": float(group.loc[minutes.lt(10 * 60), "close"].iloc[-1]) / opening - 1
            if minutes.lt(10 * 60).any() else np.nan,
            "intraday_last60_return": closing / float(group.loc[minutes.ge(15 * 60), "open"].iloc[0]) - 1
            if minutes.ge(15 * 60).any() else np.nan,
            "intraday_first30_volume_fraction": float(group.loc[minutes.lt(10 * 60), "volume"].sum()) / total_volume
            if total_volume > 0 else np.nan,
            "intraday_last60_volume_fraction": float(group.loc[minutes.ge(15 * 60), "volume"].sum()) / total_volume
            if total_volume > 0 else np.nan,
            "intraday_max_gap_minutes": float(gaps.max()) if len(gaps) else 0.0,
            "intraday_gap_over_5m_count": int(gaps.gt(5).sum()),
        })
        rows.append(base)
    return pd.DataFrame(rows)


def collect_features(events: pd.DataFrame, client: AlpacaMinuteBarsClient, cache: Path) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    grouped = events.groupby("event_date")
    for index, (date, group) in enumerate(grouped, start=1):
        target = cache / f"{pd.Timestamp(date).date()}.parquet"
        if target.exists():
            bars = pd.read_parquet(target)
        else:
            bars = client.event_day(group["symbol"].astype(str).tolist(), pd.Timestamp(date))
            bars.to_parquet(target, index=False)
            time.sleep(0.32)
        frames.append(summarize_intraday(group[["symbol", "event_date"]], bars))
        if index % 50 == 0 or index == len(grouped):
            print(f"Intraday dates collected: {index}/{len(grouped)}", flush=True)
    return pd.concat(frames, ignore_index=True)


def validation_study(base: pd.DataFrame, features: pd.DataFrame, target: str, improvement: float) -> tuple[pd.DataFrame, dict]:
    keys = ["symbol", "event_date"]
    groups = {
        "path": [column for column in features if column.startswith("intraday_") and "volume" not in column and "gap" not in column],
        "volume": [column for column in features if "volume" in column],
        "gaps": [column for column in features if "gap" in column or column == "intraday_bar_count"],
        "all_intraday": [column for column in features if column.startswith("intraday_")],
    }
    rows = []
    for model in ("random_forest", "hist_gradient_boosting"):
        baseline, _ = evaluate_variant(base, (), model, target)
        baseline["eligible"] = False
        rows.append(baseline)
        for name, columns in groups.items():
            variant = base.merge(features[[*keys, *columns]], on=keys, how="left", validate="one_to_one")
            candidate, _ = evaluate_variant(variant, (name,), model, target)
            candidate["baseline_validation_correlation"] = baseline["validation_correlation"]
            candidate["absolute_improvement"] = candidate["validation_correlation"] - baseline["validation_correlation"]
            candidate["baseline_train_validation_gap"] = baseline["train_validation_gap"]
            candidate["eligible"] = bool(
                candidate["absolute_improvement"] >= improvement
                and candidate["train_validation_gap"] <= baseline["train_validation_gap"] + 0.05
            )
            rows.append(candidate)
    results = pd.DataFrame(rows)
    eligible = results.loc[results["eligible"].fillna(False).astype(bool)]
    return results, {
        "protocol": "Train and validation intraday data only; test remains unopened unless promoted.",
        "minimum_validation_correlation_improvement": improvement,
        "eligible_variants": int(len(eligible)),
        "test_intraday_collection_authorized": bool(len(eligible)),
    }


def assessment(features: pd.DataFrame, results: pd.DataFrame, summary: dict) -> str:
    validation = features.loc[pd.to_datetime(features["event_date"]).between("2020-01-01", "2022-12-31")]
    coverage = float(validation["intraday_bar_count"].ge(30).mean()) if len(validation) else 0.0
    best = results.loc[results["combination"].ne("baseline")].sort_values("absolute_improvement", ascending=False).iloc[0]
    verdict = "PROMOTE TO LOCKED TEST COLLECTION" if summary["test_intraday_collection_authorized"] else "REJECT"
    return f"""# Point-in-Time Intraday Feature Assessment

## Verdict

**{verdict}.** SIP minute bars were collected only through the 2022 validation boundary;
test intraday data and returns were not used for selection. Validation coverage with at
least 30 bars was {coverage:.1%}. The best variant was `{best.combination}` with
{best.model}, improving validation correlation by {best.absolute_improvement:.4f} versus
the locked +0.0200 requirement.

Allocation remains zero unless a validation-eligible variant is locked before test data
is collected.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only Alpaca SIP minute-bar feature study")
    parser.add_argument("--features", default="data/processed/events_features_v5.parquet")
    parser.add_argument("--through", default="2022-12-31")
    parser.add_argument("--target", default="forward_return_5d")
    parser.add_argument("--min-improvement", type=float, default=0.02)
    parser.add_argument("--cache", default="data/raw/alpaca_intraday_train_validation")
    parser.add_argument("--output-dir", default="reports/alpaca_intraday_research")
    args = parser.parse_args()

    base = pd.read_parquet(args.features)
    base["event_date"] = pd.to_datetime(base["event_date"]).dt.normalize()
    development = base.loc[base["event_date"].le(args.through), ["symbol", "event_date"]]
    key, secret = credentials_from_environment()
    features = collect_features(development, AlpacaMinuteBarsClient(key, secret), Path(args.cache))
    results, summary = validation_study(base, features, args.target, args.min_improvement)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output / "train_validation_intraday_features.parquet", index=False)
    results.to_csv(output / "validation_results.csv", index=False)
    (output / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "ASSESSMENT.md").write_text(assessment(features, results, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
