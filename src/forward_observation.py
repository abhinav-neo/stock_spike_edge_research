from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from src.alpha_factory import build_features
from src.atomic_io import atomic_write_csv, atomic_write_json
from src.paper_trade_alpha import latest_signals


KEY_COLUMNS = ["signal_date", "candidate_key", "symbol", "direction"]
LEDGER_COLUMNS = [
    "signal_date", "candidate_rank", "candidate_key", "family", "direction", "symbol", "horizon",
    "reference_close", "avg_dollar_volume_20d", "relative_volume", "observation_status",
    "allocation_fraction", "orders_enabled", "entry_date", "entry_price", "exit_date", "exit_price",
    "gross_return", "net_return",
]


def model_fingerprint(selected_path: Path, factory_cfg: dict) -> dict[str, str]:
    selected_hash = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    config_bytes = json.dumps(factory_cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "selected_candidates_sha256": selected_hash,
        "alpha_factory_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
    }


def enforce_model_lock(lock_path: Path, fingerprint: dict[str, str], start_date: str) -> dict:
    expected = {**fingerprint, "observation_start": start_date, "model_locked": True}
    if lock_path.exists():
        actual = json.loads(lock_path.read_text(encoding="utf-8"))
        if actual != expected:
            raise RuntimeError("Locked forward model changed; preserve the existing lock or start a separately named study.")
        return actual
    atomic_write_json(expected, lock_path)
    return expected


def append_signals(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if new.empty:
        return existing.copy() if not existing.empty else pd.DataFrame(columns=LEDGER_COLUMNS)
    additions = new.copy()
    additions["signal_date"] = pd.to_datetime(additions["signal_date"]).dt.normalize()
    additions["observation_status"] = "OPEN"
    additions["allocation_fraction"] = 0.0
    additions["orders_enabled"] = False
    frames = [frame for frame in (existing, additions) if not frame.empty]
    result = pd.concat(frames, ignore_index=True, sort=False)
    result["signal_date"] = pd.to_datetime(result["signal_date"]).dt.normalize()
    return result.drop_duplicates(KEY_COLUMNS, keep="first").sort_values(KEY_COLUMNS).reset_index(drop=True)


def settle_observations(ledger: pd.DataFrame, prices: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    if ledger.empty:
        return ledger.copy()
    result = ledger.copy()
    result["signal_date"] = pd.to_datetime(result["signal_date"]).dt.normalize()
    for column in ("entry_date", "exit_date"):
        if column not in result.columns:
            result[column] = pd.NaT
        # Materialize NumPy datetime64 columns even when read_csv inferred an
        # Arrow string dtype. Arrow strings reject Timestamp assignments.
        normalized = pd.to_datetime(result[column], errors="coerce").dt.normalize()
        result[column] = pd.Series(
            normalized.to_numpy(dtype="datetime64[ns]"), index=result.index, dtype="datetime64[ns]"
        )
    for column in ("entry_price", "exit_price", "gross_return", "net_return"):
        if column not in result.columns:
            result[column] = float("nan")
        result[column] = pd.to_numeric(result[column], errors="coerce")
    bars = prices.copy()
    bars["date"] = pd.to_datetime(bars["date"]).dt.normalize()
    bars = bars.sort_values(["symbol", "date"]).drop_duplicates(["symbol", "date"], keep="last")

    for index, row in result.loc[result["observation_status"].ne("SETTLED")].iterrows():
        future = bars.loc[
            bars["symbol"].astype(str).eq(str(row["symbol"])) & bars["date"].gt(row["signal_date"])
        ]
        horizon = int(row["horizon"])
        if len(future):
            entry = future.iloc[0]
            result.at[index, "entry_date"] = entry["date"]
            result.at[index, "entry_price"] = float(entry["open"])
        if len(future) < horizon:
            continue
        exit_bar = future.iloc[horizon - 1]
        gross = float(exit_bar["close"]) / float(entry["open"]) - 1.0
        if str(row["direction"]) == "short":
            gross = -gross
        result.at[index, "exit_date"] = exit_bar["date"]
        result.at[index, "exit_price"] = float(exit_bar["close"])
        result.loc[index, "gross_return"] = gross
        result.loc[index, "net_return"] = gross - float(cost_bps) / 10_000.0
        result.loc[index, "observation_status"] = "SETTLED"
    return result


def evaluate_evidence(ledger: pd.DataFrame, gates: dict) -> dict:
    settled = ledger.loc[ledger["observation_status"].eq("SETTLED")].copy() if len(ledger) else ledger.copy()
    minimum = int(gates.get("minimum_settled_observations", 100))
    minimum_dates = int(gates.get("minimum_independent_signal_dates", 60))
    minimum_days = int(gates.get("minimum_calendar_span_days", 180))
    required_candidate_fraction = float(gates.get("minimum_positive_candidate_fraction", 0.75))
    result = {
        "settled_observations": int(len(settled)),
        "minimum_settled_observations": minimum,
        "independent_signal_dates": 0,
        "minimum_independent_signal_dates": minimum_dates,
        "calendar_span_days": 0,
        "minimum_calendar_span_days": minimum_days,
        "mean_net_return": None,
        "net_return_ci95_lower": None,
        "positive_candidate_fraction": None,
        "minimum_positive_candidate_fraction": required_candidate_fraction,
        "statistical_gate_passed": False,
        "operational_gate_passed": False,
        "breakthrough": False,
    }
    if settled.empty:
        return result

    values = pd.to_numeric(settled["net_return"], errors="coerce").dropna()
    dates = pd.to_datetime(settled["signal_date"])
    result["calendar_span_days"] = int((dates.max() - dates.min()).days)
    if values.empty:
        return result
    daily = settled.assign(
        signal_date=dates.dt.normalize(), net_return=pd.to_numeric(settled["net_return"], errors="coerce")
    ).groupby("signal_date")["net_return"].mean().dropna()
    result["independent_signal_dates"] = int(len(daily))
    mean = float(daily.mean()) if len(daily) else float(values.mean())
    standard_error = float(daily.std(ddof=1) / len(daily) ** 0.5) if len(daily) > 1 else float("inf")
    lower = mean - 1.96 * standard_error
    candidate_means = settled.assign(net_return=pd.to_numeric(settled["net_return"], errors="coerce")).groupby(
        "candidate_key"
    )["net_return"].mean()
    positive_fraction = float(candidate_means.gt(0).mean()) if len(candidate_means) else 0.0
    result.update({
        "mean_net_return": mean,
        "net_return_ci95_lower": lower if lower != float("-inf") else None,
        "positive_candidate_fraction": positive_fraction,
    })
    result["statistical_gate_passed"] = bool(
        len(values) >= minimum
        and len(daily) >= minimum_dates
        and result["calendar_span_days"] >= minimum_days
        and lower > 0
        and positive_fraction >= required_candidate_fraction
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Record zero-capital forward observations; never creates orders.")
    parser.add_argument("--config", default="config/alpha_factory.yaml")
    parser.add_argument("--prices", default="data/processed/daily_prices.parquet")
    parser.add_argument("--selected", default="reports/alpha_portfolio_selected_candidates.csv")
    parser.add_argument("--ledger", default="reports/forward_observation/ledger.csv")
    parser.add_argument("--manifest", default="reports/forward_observation/manifest.json")
    parser.add_argument("--model-lock", default="reports/forward_observation/model_lock.json")
    parser.add_argument("--assessment", default="reports/forward_observation/assessment.json")
    args = parser.parse_args()

    root = yaml.safe_load(Path(args.config).read_text())
    prices = pd.read_parquet(args.prices)
    prices["date"] = pd.to_datetime(prices["date"])
    selected_path = Path(args.selected)
    selected = pd.read_csv(selected_path)
    factory_cfg = root.get("alpha_factory", {})
    signals = latest_signals(build_features(prices), selected, factory_cfg)
    observation_cfg = root.get("forward_observation", {})
    observation_start = pd.Timestamp(observation_cfg["start_date"]).normalize()
    lock = enforce_model_lock(
        Path(args.model_lock), model_fingerprint(selected_path, factory_cfg), str(observation_start.date())
    )
    if not signals.empty:
        signals = signals.loc[pd.to_datetime(signals["signal_date"]).ge(observation_start)].copy()
    ledger_path = Path(args.ledger)
    existing = pd.read_csv(ledger_path) if ledger_path.exists() and ledger_path.stat().st_size else pd.DataFrame()
    ledger = append_signals(existing, signals)
    ledger = settle_observations(ledger, prices, root.get("alpha_factory", {}).get("round_trip_cost_bps", 100.0))
    atomic_write_csv(ledger, ledger_path)

    assessment = evaluate_evidence(ledger, observation_cfg.get("gates", {}))
    assessment_path = Path(args.assessment)
    atomic_write_json(assessment, assessment_path)

    manifest = {
        "data_cutoff": str(prices["date"].max().date()),
        "observation_start": str(observation_start.date()),
        "observations": int(len(ledger)),
        "open": int(ledger["observation_status"].eq("OPEN").sum()) if len(ledger) else 0,
        "settled": int(ledger["observation_status"].eq("SETTLED").sum()) if len(ledger) else 0,
        "allocation_fraction": 0.0,
        "orders_enabled": False,
        "model_locked": True,
        "statistical_gate_passed": assessment["statistical_gate_passed"],
        "operational_gate_passed": assessment["operational_gate_passed"],
        "breakthrough": assessment["breakthrough"],
        **{key: lock[key] for key in ("selected_candidates_sha256", "alpha_factory_config_sha256")},
    }
    manifest_path = Path(args.manifest)
    atomic_write_json(manifest, manifest_path)
    print(pd.DataFrame([manifest]).to_string(index=False))


if __name__ == "__main__":
    main()
