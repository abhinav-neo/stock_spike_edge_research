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


PARAMETER_COLUMNS = list(TrialConfig.__dataclass_fields__)


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


def safe_number(value: Any, default: float = float("nan")) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result


def inclusive_years(start: pd.Timestamp, end: pd.Timestamp) -> float:
    days = max((end - start).days + 1, 1)
    return days / 365.25


def validation_period_lengths(base_cfg: dict[str, Any], events: pd.DataFrame) -> tuple[float, float]:
    dates = pd.to_datetime(events["event_date"], errors="coerce").dropna()
    if dates.empty:
        return 1.0, 1.0

    train_end = pd.Timestamp(base_cfg["train_end"])
    validation_start = pd.Timestamp(base_cfg["validation_start"])
    validation_end = pd.Timestamp(base_cfg["validation_end"])
    train_start = min(dates.min(), train_end)
    return (
        inclusive_years(train_start, train_end),
        inclusive_years(validation_start, validation_end),
    )


def frequency_adjusted_return_proxy(
    mean_trade_return: float,
    horizon: int,
    train_n: int,
    validation_n: int,
    train_years: float,
    validation_years: float,
) -> tuple[float, float]:
    """Estimate an annual return proxy using observed opportunity frequency.

    This remains a screening proxy, not portfolio CAGR. It uses the lower annual
    signal frequency from train and validation and caps turnover by holding period.
    """
    if not np.isfinite(mean_trade_return) or mean_trade_return <= -1:
        return float("nan"), 0.0

    train_frequency = train_n / max(train_years, 1e-9)
    validation_frequency = validation_n / max(validation_years, 1e-9)
    conservative_frequency = min(train_frequency, validation_frequency)
    holding_period_capacity = 252.0 / max(horizon, 1)
    annual_opportunities = max(min(conservative_frequency, holding_period_capacity), 0.0)

    try:
        proxy = float((1.0 + mean_trade_return) ** annual_opportunities - 1.0)
    except OverflowError:
        proxy = float("inf")
    return proxy, annual_opportunities


def score_row(
    row: pd.Series,
    target_cagr: float,
    train_years: float,
    validation_years: float,
) -> dict[str, Any]:
    train_mean = safe_number(row.get("train_mean_return"))
    validation_mean = safe_number(row.get("validation_mean_return"))
    train_pf = safe_number(row.get("train_profit_factor"), 0.0)
    validation_pf = safe_number(row.get("validation_profit_factor"), 0.0)
    train_t = safe_number(row.get("train_t_stat"), 0.0)
    validation_t = safe_number(row.get("validation_t_stat"), 0.0)
    train_n = int(safe_number(row.get("train_n"), 0.0))
    validation_n = int(safe_number(row.get("validation_n"), 0.0))
    horizon = int(row["horizon"])

    conservative_mean = min(train_mean, validation_mean)
    annualized_proxy, annual_opportunities = frequency_adjusted_return_proxy(
        conservative_mean,
        horizon,
        train_n,
        validation_n,
        train_years,
        validation_years,
    )

    sample_size_pass = bool(row.get("sample_size_pass", False))
    positive = bool(
        sample_size_pass
        and np.isfinite(conservative_mean)
        and train_mean > 0
        and validation_mean > 0
    )
    target_pass = bool(positive and np.isfinite(annualized_proxy) and annualized_proxy >= target_cagr)

    if not sample_size_pass:
        score = -math.inf
        rejection_reason = "minimum_sample_size"
    elif not positive:
        score = -math.inf
        rejection_reason = "non_positive_train_or_validation"
    else:
        pf_component = max(min(train_pf, validation_pf, 5.0), 0.0)
        t_component = max(min(train_t, validation_t, 5.0), -5.0)
        return_component = math.log1p(max(annualized_proxy, -0.999))
        score = (
            2.0 * return_component
            + 0.50 * pf_component
            + 0.25 * t_component
            + (1.0 if target_pass else 0.0)
        )
        rejection_reason = ""

    return {
        "optimizer_score": score,
        "conservative_mean_trade_return": conservative_mean,
        "annualized_return_proxy": annualized_proxy,
        "annual_opportunities_proxy": annual_opportunities,
        "target_proxy_pass": target_pass,
        "eligible_for_ranking": positive,
        "rejection_reason": rejection_reason,
        "ranking_uses_test_period": False,
    }


def evaluate_trial(
    events: pd.DataFrame,
    base_cfg: dict[str, Any],
    trial: TrialConfig,
    target_cagr: float,
    train_years: float,
    validation_years: float,
) -> dict[str, Any]:
    result = evaluate(events, trial_validation_config(base_cfg, trial))
    selected = result[result["side"] == trial.side]
    if selected.empty:
        return {
            **asdict(trial),
            "status": "no_result",
            "optimizer_score": -math.inf,
            "eligible_for_ranking": False,
            "rejection_reason": "no_result",
        }

    row = selected.iloc[0]
    output = {**asdict(trial), "status": "ok"}
    output.update({key: value for key, value in row.to_dict().items() if key != "rule"})
    output.update(score_row(row, target_cagr, train_years, validation_years))
    return output


def load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return pd.DataFrame()
    return deduplicate_trials(frame)


def normalized_parameter_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in PARAMETER_COLUMNS:
        if column not in normalized:
            continue
        if column in {"side"}:
            normalized[column] = normalized[column].astype(str)
        elif column == "horizon":
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").astype("Int64")
        else:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce").round(10)
    return normalized


def deduplicate_trials(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not set(PARAMETER_COLUMNS).issubset(frame.columns):
        return frame
    normalized = normalized_parameter_frame(frame)
    return normalized.drop_duplicates(subset=PARAMETER_COLUMNS, keep="last").reset_index(drop=True)


def trial_key(trial: TrialConfig) -> str:
    payload = asdict(trial)
    for key, value in payload.items():
        if isinstance(value, float):
            payload[key] = round(value, 10)
    return json.dumps(payload, sort_keys=True)


def existing_keys(frame: pd.DataFrame) -> set[str]:
    if frame.empty or not set(PARAMETER_COLUMNS).issubset(frame.columns):
        return set()
    normalized = normalized_parameter_frame(frame)
    keys: set[str] = set()
    for _, row in normalized.iterrows():
        payload: dict[str, Any] = {}
        for name in PARAMETER_COLUMNS:
            value = row[name]
            if name == "horizon":
                payload[name] = int(value)
            elif name == "side":
                payload[name] = str(value)
            else:
                payload[name] = round(float(value), 10)
        keys.add(json.dumps(payload, sort_keys=True))
    return keys


def write_outputs(frame: pd.DataFrame, output_dir: Path, target_cagr: float) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    clean = deduplicate_trials(frame)
    clean.to_csv(output_dir / "hyperparameter_trials.csv", index=False)

    eligible_mask = clean.get(
        "eligible_for_ranking", pd.Series(False, index=clean.index)
    ).fillna(False).astype(bool)
    eligible = clean[eligible_mask].sort_values(
        "optimizer_score", ascending=False, na_position="last"
    )
    rejected = clean[~eligible_mask].copy()
    if not rejected.empty:
        rejected = rejected.sort_values(
            ["rejection_reason", "train_n", "validation_n"],
            ascending=[True, False, False],
            na_position="last",
        )

    eligible.head(100).to_csv(output_dir / "hyperparameter_leaderboard.csv", index=False)
    rejected.to_csv(output_dir / "rejected_candidates.csv", index=False)

    survivors = eligible[
        eligible.get("target_proxy_pass", pd.Series(False, index=eligible.index)).fillna(False)
    ]
    survivors.to_csv(output_dir / "target_proxy_survivors.csv", index=False)

    summary = {
        "unique_trials_completed": int(len(clean)),
        "eligible_candidates": int(len(eligible)),
        "rejected_candidates": int(len(rejected)),
        "target_cagr": target_cagr,
        "target_proxy_survivors": int(len(survivors)),
        "best_optimizer_score": safe_number(eligible.iloc[0].get("optimizer_score"))
        if not eligible.empty
        else None,
        "best_annualized_return_proxy": safe_number(
            eligible.iloc[0].get("annualized_return_proxy")
        )
        if not eligible.empty
        else None,
        "warning": (
            "annualized_return_proxy uses observed signal frequency but is still not "
            "portfolio CAGR; validate finalists with chronological portfolio simulation, "
            "execution constraints, and Monte Carlo analysis"
        ),
    }
    (output_dir / "hyperparameter_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return eligible


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
    train_years, validation_years = validation_period_lengths(base_cfg, events)
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
        records.append(
            evaluate_trial(
                events,
                base_cfg,
                trial,
                args.target_cagr,
                train_years,
                validation_years,
            )
        )

        if len(records) % args.checkpoint_every == 0:
            eligible = write_outputs(pd.DataFrame(records), output_dir, args.target_cagr)
            print(
                f"Completed {len(records)}/{args.trials} unique trials; "
                f"{len(eligible)} eligible"
            )

    frame = pd.DataFrame(records)
    eligible = write_outputs(frame, output_dir, args.target_cagr)

    if len(records) < args.trials:
        print(
            "Stopped after exhausting the available sampled combinations: "
            f"{len(records)} unique trials"
        )
    else:
        print(f"Completed {len(records)} unique trials")

    display_columns = [
        "side",
        "return_low",
        "return_high",
        "close_location",
        "relative_volume",
        "minimum_price",
        "horizon",
        "optimizer_score",
        "conservative_mean_trade_return",
        "annual_opportunities_proxy",
        "annualized_return_proxy",
        "target_proxy_pass",
        "train_n",
        "train_mean_return",
        "train_profit_factor",
        "train_t_stat",
        "validation_n",
        "validation_mean_return",
        "validation_profit_factor",
        "validation_t_stat",
        "test_n",
        "test_mean_return",
        "test_profit_factor",
        "test_t_stat",
        "train_worst_trade",
        "validation_worst_trade",
        "test_worst_trade",
    ]
    print("\nEligible candidates only (test metrics shown but excluded from ranking)")
    if eligible.empty:
        print("No candidate passed sample-size and train/validation positivity requirements.")
    else:
        print(eligible[[c for c in display_columns if c in eligible]].head(20).to_string(index=False))
    print(f"\nOutputs written to: {output_dir}")


if __name__ == "__main__":
    main()
