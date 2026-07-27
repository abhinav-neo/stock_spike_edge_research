from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.analyze_edges import evaluate


@dataclass(frozen=True)
class TrialConfig:
    side: str
    return_low: float
    return_high: float
    close_location: float
    relative_volume: float
    minimum_price: float
    horizon: int


def parse_float_list(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_bands(value: str) -> list[tuple[float, float]]:
    bands: list[tuple[float, float]] = []
    for item in value.split(","):
        lo_text, hi_text = item.strip().split(":", maxsplit=1)
        lo, hi = float(lo_text), float(hi_text)
        if lo >= hi:
            raise ValueError(f"Invalid return band {item!r}: low must be below high")
        bands.append((lo, hi))
    return bands


def sample_trial(rng: random.Random, args: argparse.Namespace) -> TrialConfig:
    side = rng.choice(args.sides)
    return_low, return_high = rng.choice(args.return_bands)
    close_location = rng.choice(
        args.continuation_close_locations
        if side == "long"
        else args.failed_spike_close_locations
    )
    return TrialConfig(
        side=side,
        return_low=return_low,
        return_high=return_high,
        close_location=close_location,
        relative_volume=rng.choice(args.relative_volumes),
        minimum_price=rng.choice(args.minimum_prices),
        horizon=rng.choice(args.horizons),
    )


def trial_validation_config(base_cfg: dict[str, Any], trial: TrialConfig) -> dict[str, Any]:
    cfg = dict(base_cfg)
    cfg["horizons"] = [trial.horizon]
    cfg["parameter_grid"] = {
        "continuation_return_bands": [[trial.return_low, trial.return_high]],
        "continuation_close_locations": [trial.close_location],
        "failed_spike_close_locations": [trial.close_location],
        "relative_volumes": [trial.relative_volume],
        "minimum_prices": [trial.minimum_price],
    }
    return cfg


def annualized_return_proxy(mean_trade_return: float, horizon: int) -> float:
    """Convert mean trade return to a comparable annualized proxy.

    This is not portfolio CAGR because the event study does not model overlapping
    positions, cash usage, exposure, or capacity. It is used only for ranking.
    """
    if not np.isfinite(mean_trade_return) or mean_trade_return <= -1:
        return float("nan")
    periods = 252.0 / max(horizon, 1)
    try:
        return float((1.0 + mean_trade_return) ** periods - 1.0)
    except OverflowError:
        return float("inf")


def safe_number(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def score_row(row: pd.Series, target_cagr: float) -> dict[str, Any]:
    train_mean = safe_number(row.get("train_mean_return"))
    validation_mean = safe_number(row.get("validation_mean_return"))
    train_pf = safe_number(row.get("train_profit_factor"), 0.0)
    validation_pf = safe_number(row.get("validation_profit_factor"), 0.0)
    train_t = safe_number(row.get("train_t_stat"), 0.0)
    validation_t = safe_number(row.get("validation_t_stat"), 0.0)
    horizon = int(row["horizon"])

    conservative_mean = min(train_mean, validation_mean)
    annualized_proxy = annualized_return_proxy(conservative_mean, horizon)

    positive = bool(
        row.get("sample_size_pass", False)
        and np.isfinite(conservative_mean)
        and conservative_mean > 0
    )
    target_pass = bool(positive and annualized_proxy >= target_cagr)

    # Rank only with train and validation. The untouched test metrics are retained
    # in the output for later inspection but never influence this score.
    pf_component = min(train_pf, validation_pf, 5.0)
    t_component = min(train_t, validation_t, 5.0)
    return_component = math.log1p(max(annualized_proxy, -0.999)) if np.isfinite(annualized_proxy) else -10.0
    score = (
        2.0 * return_component
        + 0.50 * pf_component
        + 0.25 * t_component
        + (1.0 if target_pass else 0.0)
        + (1.0 if positive else -2.0)
    )

    return {
        "optimizer_score": score,
        "conservative_mean_trade_return": conservative_mean,
        "annualized_return_proxy": annualized_proxy,
        "target_proxy_pass": target_pass,
        "ranking_uses_test_period": False,
    }


def evaluate_trial(
    events: pd.DataFrame,
    base_cfg: dict[str, Any],
    trial: TrialConfig,
    target_cagr: float,
) -> dict[str, Any]:
    result = evaluate(events, trial_validation_config(base_cfg, trial))
    selected = result[result["side"] == trial.side]
    if selected.empty:
        return {
            **asdict(trial),
            "status": "no_result",
            "optimizer_score": -math.inf,
        }

    row = selected.iloc[0]
    output = {**asdict(trial), "status": "ok"}
    output.update({key: value for key, value in row.to_dict().items() if key != "rule"})
    output.update(score_row(row, target_cagr))
    return output


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    return frame if not frame.empty else pd.DataFrame()


def trial_key(trial: TrialConfig) -> str:
    return json.dumps(asdict(trial), sort_keys=True)


def existing_keys(frame: pd.DataFrame) -> set[str]:
    required = set(TrialConfig.__dataclass_fields__)
    if frame.empty or not required.issubset(frame.columns):
        return set()
    return {
        json.dumps({name: row[name] for name in required}, sort_keys=True)
        for _, row in frame.iterrows()
    }


def write_outputs(frame: pd.DataFrame, output_dir: Path, target_cagr: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = frame.sort_values("optimizer_score", ascending=False, na_position="last")
    ranked.to_csv(output_dir / "hyperparameter_trials.csv", index=False)
    ranked.head(100).to_csv(output_dir / "hyperparameter_leaderboard.csv", index=False)

    survivors = ranked[
        ranked.get("target_proxy_pass", pd.Series(False, index=ranked.index)).fillna(False)
    ]
    survivors.to_csv(output_dir / "target_proxy_survivors.csv", index=False)

    summary = {
        "trials_completed": int(len(ranked)),
        "target_cagr": target_cagr,
        "target_proxy_survivors": int(len(survivors)),
        "best_optimizer_score": safe_number(ranked.iloc[0].get("optimizer_score"))
        if not ranked.empty
        else None,
        "best_annualized_return_proxy": safe_number(
            ranked.iloc[0].get("annualized_return_proxy")
        )
        if not ranked.empty
        else None,
        "warning": (
            "annualized_return_proxy is not portfolio CAGR; validate finalists with "
            "the historical portfolio simulator and Monte Carlo analysis"
        ),
    }
    (output_dir / "hyperparameter_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Resumable random search over the existing edge-analysis parameter space"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--events", default="data/processed/events.parquet")
    parser.add_argument("--output-dir", default="reports/hyperparameter_optimizer")
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--target-cagr", type=float, default=0.24)
    parser.add_argument("--sides", type=lambda x: x.split(","), default=["long", "short"])
    parser.add_argument(
        "--return-bands",
        type=parse_bands,
        default=parse_bands("0.40:0.60,0.60:1.00,1.00:10.00"),
    )
    parser.add_argument(
        "--continuation-close-locations",
        type=parse_float_list,
        default=parse_float_list("0.50,0.60,0.70,0.75,0.80,0.90"),
    )
    parser.add_argument(
        "--failed-spike-close-locations",
        type=parse_float_list,
        default=parse_float_list("0.15,0.20,0.25,0.30,0.40,0.50,0.60"),
    )
    parser.add_argument(
        "--relative-volumes",
        type=parse_float_list,
        default=parse_float_list("1.5,2,3,5,7.5,10"),
    )
    parser.add_argument(
        "--minimum-prices",
        type=parse_float_list,
        default=parse_float_list("1,3,5,10,20"),
    )
    parser.add_argument(
        "--horizons",
        type=parse_int_list,
        default=parse_int_list("1,2,3,5,10,20,40,60"),
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    if not set(args.sides).issubset({"long", "short"}):
        raise ValueError("--sides may contain only long and short")

    base_cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["validation"]
    events = pd.read_parquet(args.events)
    output_dir = Path(args.output_dir)
    trials_path = output_dir / "hyperparameter_trials.csv"

    prior = pd.DataFrame() if args.no_resume else load_existing(trials_path)
    completed = existing_keys(prior)
    records = prior.to_dict("records") if not prior.empty else []
    rng = random.Random(args.seed)

    attempts = 0
    max_attempts = max(args.trials * 50, 1000)
    while len(records) < args.trials and attempts < max_attempts:
        attempts += 1
        trial = sample_trial(rng, args)
        key = trial_key(trial)
        if key in completed:
            continue
        completed.add(key)
        records.append(evaluate_trial(events, base_cfg, trial, args.target_cagr))

        if len(records) % args.checkpoint_every == 0:
            write_outputs(pd.DataFrame(records), output_dir, args.target_cagr)
            print(f"Completed {len(records)}/{args.trials} unique trials")

    frame = pd.DataFrame(records)
    write_outputs(frame, output_dir, args.target_cagr)

    if len(records) < args.trials:
        print(
            f"Stopped after exhausting the available sampled combinations: "
            f"{len(records)} unique trials"
        )
    else:
        print(f"Completed {len(records)} unique trials")

    ranked = frame.sort_values("optimizer_score", ascending=False, na_position="last")
    display_columns = [
        "side",
        "return_low",
        "return_high",
        "close_location",
        "relative_volume",
        "minimum_price",
        "horizon",
        "optimizer_score",
        "annualized_return_proxy",
        "target_proxy_pass",
        "train_n",
        "validation_n",
        "test_n",
    ]
    print("\nTop candidates (test metrics shown but excluded from ranking)")
    print(ranked[[c for c in display_columns if c in ranked]].head(20).to_string(index=False))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
