from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REQUIRED_COLUMNS = {"symbol", "date", "open", "high", "low", "close", "volume"}


def validate_prices(prices: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(prices.columns)
    if missing:
        raise ValueError(f"daily price data missing required columns: {sorted(missing)}")
    if prices.duplicated(["symbol", "date"]).any():
        raise ValueError("daily price data contains duplicate symbol/date rows")


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    validate_prices(prices)
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["symbol", "date"]).reset_index(drop=True)
    group = frame.groupby("symbol", group_keys=False)

    frame["previous_close"] = group["close"].shift(1)
    frame["return_1d"] = group["close"].pct_change()
    frame["return_5d"] = group["close"].pct_change(5)
    frame["return_20d"] = group["close"].pct_change(20)
    frame["gap_return"] = frame["open"] / frame["previous_close"] - 1.0
    frame["intraday_return"] = frame["close"] / frame["open"] - 1.0
    frame["range_pct"] = (frame["high"] - frame["low"]) / frame["previous_close"]
    denominator = (frame["high"] - frame["low"]).replace(0, np.nan)
    frame["close_location"] = (frame["close"] - frame["low"]) / denominator
    frame["dollar_volume"] = frame["close"] * frame["volume"]
    frame["avg_dollar_volume_20d"] = group["dollar_volume"].transform(lambda s: s.shift(1).rolling(20).mean())
    frame["avg_volume_20d"] = group["volume"].transform(lambda s: s.shift(1).rolling(20).mean())
    frame["relative_volume"] = frame["volume"] / frame["avg_volume_20d"]
    frame["ma_20"] = group["close"].transform(lambda s: s.rolling(20).mean())
    frame["ma_50"] = group["close"].transform(lambda s: s.rolling(50).mean())
    frame["distance_ma20"] = frame["close"] / frame["ma_20"] - 1.0
    frame["distance_ma50"] = frame["close"] / frame["ma_50"] - 1.0
    frame["rolling_high_20"] = group["high"].transform(lambda s: s.shift(1).rolling(20).max())
    frame["breakout_20"] = frame["close"] / frame["rolling_high_20"] - 1.0
    frame["rolling_low_20"] = group["low"].transform(lambda s: s.shift(1).rolling(20).min())
    frame["breakdown_20"] = frame["close"] / frame["rolling_low_20"] - 1.0
    return frame


def _forward_return(frame: pd.DataFrame, horizon: int, direction: str) -> pd.Series:
    future = frame.groupby("symbol")["close"].shift(-int(horizon)) / frame["close"] - 1.0
    return future if direction == "long" else -future


def candidate_specs(config: dict) -> list[dict]:
    specs: list[dict] = []
    families = config.get("families", {})

    for direction, gaps, rvols, close_locations, horizons in itertools.product(
        ["short", "long"],
        families.get("gap_fade", {}).get("gap_thresholds", [0.05, 0.10]),
        families.get("gap_fade", {}).get("relative_volumes", [1.5, 3.0]),
        families.get("gap_fade", {}).get("close_locations", [0.25, 0.75]),
        families.get("gap_fade", {}).get("horizons", [1, 2, 5]),
    ):
        condition = "gap_up_fade" if direction == "short" else "gap_down_fade"
        specs.append({"family": "gap_fade", "direction": direction, "condition": condition, "gap": gaps, "rvol": rvols, "close_location": close_locations, "horizon": horizons})

    for direction, returns, rvols, distances, horizons in itertools.product(
        ["long", "short"],
        families.get("momentum", {}).get("return_thresholds", [0.10, 0.20]),
        families.get("momentum", {}).get("relative_volumes", [1.5, 3.0]),
        families.get("momentum", {}).get("ma20_distances", [0.05, 0.10]),
        families.get("momentum", {}).get("horizons", [5, 10, 20]),
    ):
        specs.append({"family": "momentum", "direction": direction, "return_threshold": returns, "rvol": rvols, "ma20_distance": distances, "horizon": horizons})

    for direction, distances, rvols, horizons in itertools.product(
        ["long", "short"],
        families.get("mean_reversion", {}).get("ma20_distances", [0.10, 0.20]),
        families.get("mean_reversion", {}).get("relative_volumes", [1.0, 2.0]),
        families.get("mean_reversion", {}).get("horizons", [2, 5, 10]),
    ):
        specs.append({"family": "mean_reversion", "direction": direction, "ma20_distance": distances, "rvol": rvols, "horizon": horizons})

    for direction, breakouts, rvols, horizons in itertools.product(
        ["long", "short"],
        families.get("breakout", {}).get("breakout_thresholds", [0.0, 0.02]),
        families.get("breakout", {}).get("relative_volumes", [1.5, 3.0]),
        families.get("breakout", {}).get("horizons", [5, 10, 20]),
    ):
        specs.append({"family": "breakout", "direction": direction, "breakout_threshold": breakouts, "rvol": rvols, "horizon": horizons})
    return specs


def candidate_mask(frame: pd.DataFrame, spec: dict, cfg: dict) -> pd.Series:
    mask = (
        frame["close"].ge(float(cfg.get("minimum_price", 3.0)))
        & frame["avg_dollar_volume_20d"].ge(float(cfg.get("minimum_avg_dollar_volume", 2_000_000)))
        & frame["relative_volume"].ge(float(spec.get("rvol", 0.0)))
    )
    family = spec["family"]
    direction = spec["direction"]

    if family == "gap_fade":
        gap = float(spec["gap"])
        location = float(spec["close_location"])
        if direction == "short":
            mask &= frame["gap_return"].ge(gap) & frame["close_location"].le(location)
        else:
            mask &= frame["gap_return"].le(-gap) & frame["close_location"].ge(location)
    elif family == "momentum":
        threshold = float(spec["return_threshold"])
        distance = float(spec["ma20_distance"])
        if direction == "long":
            mask &= frame["return_5d"].ge(threshold) & frame["distance_ma20"].ge(distance)
        else:
            mask &= frame["return_5d"].le(-threshold) & frame["distance_ma20"].le(-distance)
    elif family == "mean_reversion":
        distance = float(spec["ma20_distance"])
        if direction == "long":
            mask &= frame["distance_ma20"].le(-distance)
        else:
            mask &= frame["distance_ma20"].ge(distance)
    elif family == "breakout":
        threshold = float(spec["breakout_threshold"])
        if direction == "long":
            mask &= frame["breakout_20"].ge(threshold)
        else:
            mask &= frame["breakdown_20"].le(-threshold)
    else:
        raise ValueError(f"unsupported family: {family}")
    return mask.fillna(False)


def evaluate_candidate(frame: pd.DataFrame, spec: dict, cfg: dict) -> dict:
    mask = candidate_mask(frame, spec, cfg)
    forward = _forward_return(frame, int(spec["horizon"]), spec["direction"])
    sample = forward.loc[mask].dropna()
    cost = float(cfg.get("round_trip_cost_bps", 100.0)) / 10000.0
    net = sample - cost
    n = int(len(net))
    mean = float(net.mean()) if n else np.nan
    median = float(net.median()) if n else np.nan
    std = float(net.std(ddof=1)) if n > 1 else np.nan
    t_stat = mean / (std / np.sqrt(n)) if n > 1 and std > 0 else np.nan
    win_rate = float((net > 0).mean()) if n else np.nan
    years = frame.loc[mask, "date"].dt.year
    yearly = pd.DataFrame({"year": years, "net": net}).dropna().groupby("year")["net"].mean()
    positive_year_fraction = float((yearly > 0).mean()) if len(yearly) else np.nan
    min_events = int(cfg.get("minimum_events", 100))
    score = 0.0
    if n:
        score = (
            max(mean, -1.0) * 100.0
            + max(win_rate - 0.5, -0.5) * 20.0
            + (t_stat if np.isfinite(t_stat) else -10.0)
            + (positive_year_fraction if np.isfinite(positive_year_fraction) else 0.0) * 5.0
            - max(min_events - n, 0) / max(min_events, 1) * 10.0
        )
    candidate_id = f"{spec['family']}|{spec['direction']}|h{spec['horizon']}|{json.dumps(spec, sort_keys=True)}"
    return {
        "candidate_id": candidate_id,
        **spec,
        "events": n,
        "mean_net_return": mean,
        "median_net_return": median,
        "win_rate": win_rate,
        "t_stat": t_stat,
        "years": int(len(yearly)),
        "positive_year_fraction": positive_year_fraction,
        "minimum_events_pass": n >= min_events,
        "positive_expectancy_pass": bool(np.isfinite(mean) and mean > 0),
        "initial_research_pass": bool(n >= min_events and np.isfinite(mean) and mean > 0 and np.isfinite(t_stat) and t_stat >= float(cfg.get("minimum_t_stat", 2.0))),
        "research_score": float(score),
        "production_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/alpha_factory.yaml")
    parser.add_argument("--prices", default="data/processed/daily_prices.parquet")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    factory_cfg = cfg.get("alpha_factory", {})
    prices = pd.read_parquet(args.prices)
    features = build_features(prices)
    specs = candidate_specs(factory_cfg)
    rows = [evaluate_candidate(features, spec, factory_cfg) for spec in specs]
    result = pd.DataFrame(rows).sort_values(["initial_research_pass", "research_score"], ascending=[False, False])

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "alpha_factory_candidates.csv", index=False)
    result.head(int(factory_cfg.get("top_n", 50))).to_csv(output / "alpha_factory_top_candidates.csv", index=False)

    summary = result.groupby("family", as_index=False).agg(
        candidates=("candidate_id", "size"),
        initial_passes=("initial_research_pass", "sum"),
        best_score=("research_score", "max"),
        best_mean_net_return=("mean_net_return", "max"),
    )
    summary.to_csv(output / "alpha_factory_family_summary.csv", index=False)

    display = ["family", "direction", "horizon", "events", "mean_net_return", "win_rate", "t_stat", "positive_year_fraction", "initial_research_pass", "research_score"]
    print(result[display].head(int(factory_cfg.get("top_n", 50))).to_string(index=False))
    print(f"\nCandidates generated: {len(result)}")
    print(f"Initial research passes: {int(result['initial_research_pass'].sum())}")
    print("Production-approved candidates: 0")


if __name__ == "__main__":
    main()
