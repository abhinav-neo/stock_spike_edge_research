from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from src.atomic_io import atomic_write_json
from src.forward_observation import evaluate_evidence


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def combined_verdict(
    statistical: dict,
    ledger: pd.DataFrame,
    snapshots: pd.DataFrame,
    executions: pd.DataFrame,
    gates: dict,
    accounts: pd.DataFrame | None = None,
    eligibility: pd.DataFrame | None = None,
    locates: pd.DataFrame | None = None,
) -> dict:
    settled = ledger.loc[ledger["observation_status"].eq("SETTLED")].copy() if len(ledger) else ledger
    eligibility = eligibility if eligibility is not None else pd.DataFrame()
    locates = locates if locates is not None else pd.DataFrame()
    eligible_ids = set(
        eligibility.loc[eligibility["broker_eligible"].fillna(False).astype(bool), "observation_id"].astype(str)
    ) if len(eligibility) else set()
    settled_ids = set()
    if len(settled):
        from src.forward_quote_capture import observation_id
        settled_ids = {observation_id(row) for _, row in settled.iterrows()}
    eligible_settled_ids = settled_ids & eligible_ids
    execution_ids = set(executions["observation_id"].astype(str)) if len(executions) else set()
    ledger_ids = set()
    duplicate_observations = 0
    if len(ledger):
        from src.forward_quote_capture import observation_id
        generated_ids = ledger.apply(observation_id, axis=1)
        ledger_ids = set(generated_ids)
        duplicate_observations = int(generated_ids.duplicated().sum())
    eligibility_ids = set(eligibility["observation_id"].astype(str)) if len(eligibility) else set()
    unknown_eligibility_ids = eligibility_ids - ledger_ids
    rejected_execution_ids = execution_ids - eligible_ids
    short_ids = set()
    if len(ledger):
        from src.forward_quote_capture import observation_id
        short_ids = {
            observation_id(row) for _, row in ledger.loc[ledger["direction"].eq("short")].iterrows()
        }
    required_locate_ids = eligible_ids & short_ids
    locate_ids = set(locates["observation_id"].astype(str)) if len(locates) else set()
    confirmed_locate_ids = set()
    if len(locates):
        confirmed_locate_ids = set(
            locates.loc[locates["locate_confirmed"].fillna(False).astype(bool), "observation_id"].astype(str)
        )
    unknown_locate_ids = locate_ids - ledger_ids
    integrity_passed = bool(
        duplicate_observations == 0
        and eligibility_ids == ledger_ids
        and not unknown_eligibility_ids
        and not rejected_execution_ids
        and not unknown_locate_ids
    )
    execution_coverage = (
        float(len(execution_ids & eligible_settled_ids) / len(eligible_settled_ids)) if eligible_settled_ids else 0.0
    )
    eligibility_coverage = float(len(eligibility) / len(ledger)) if len(ledger) else 0.0
    snapshot_keys = set()
    if len(snapshots):
        snapshot_keys = set(zip(snapshots["signal_date"].astype(str), snapshots["symbol"].astype(str)))
    required_keys = set(zip(settled["signal_date"].astype(str), settled["symbol"].astype(str))) if len(settled) else set()
    snapshot_coverage = float(len(required_keys & snapshot_keys) / len(required_keys)) if required_keys else 0.0

    short_metadata = snapshots.loc[snapshots["direction"].eq("short")].copy() if len(snapshots) else snapshots
    shortable_fraction = float(short_metadata["shortable"].fillna(False).astype(bool).mean()) if len(short_metadata) else 0.0
    easy_fraction = (
        float(short_metadata["easy_to_borrow"].fillna(False).astype(bool).mean()) if len(short_metadata) else 0.0
    )
    eligible_rows = eligibility.loc[eligibility["broker_eligible"].fillna(False).astype(bool)] if len(eligibility) else eligibility
    eligible_shortable_fraction = (
        float(eligible_rows["shortable"].fillna(False).astype(bool).mean()) if len(eligible_rows) else 0.0
    )
    eligible_easy_fraction = (
        float(eligible_rows["easy_to_borrow"].fillna(False).astype(bool).mean()) if len(eligible_rows) else 0.0
    )
    spread_values = []
    if len(executions):
        spread_values = pd.concat([
            pd.to_numeric(executions["entry_spread_bps"], errors="coerce"),
            pd.to_numeric(executions["exit_spread_bps"], errors="coerce"),
        ]).dropna()
    median_spread = float(spread_values.median()) if len(spread_values) else None
    locate_coverage = (
        float(len(required_locate_ids & confirmed_locate_ids) / len(required_locate_ids))
        if required_locate_ids else 0.0
    )

    minimum_execution = float(gates.get("minimum_execution_coverage", 0.95))
    minimum_snapshot = float(gates.get("minimum_snapshot_coverage", 0.95))
    minimum_shortable = float(gates.get("minimum_shortable_fraction", 1.0))
    minimum_easy = float(gates.get("minimum_easy_to_borrow_fraction", 0.95))
    maximum_spread = float(gates.get("maximum_median_spread_bps", 50.0))
    require_locates = bool(gates.get("require_actual_locates", True))
    account_ready = False
    if accounts is not None and len(accounts):
        account = accounts.sort_values("snapshot_date").iloc[-1]
        account_ready = bool(
            str(account.get("status", "")).upper() == "ACTIVE"
            and bool(account.get("shorting_enabled", False))
            and not bool(account.get("trading_blocked", True))
            and not bool(account.get("account_blocked", True))
            and not bool(account.get("trade_suspended_by_user", True))
        )
    operational = bool(
        len(eligible_settled_ids) > 0
        and integrity_passed
        and eligibility_coverage >= 1.0
        and execution_coverage >= minimum_execution
        and snapshot_coverage >= minimum_snapshot
        and eligible_shortable_fraction >= minimum_shortable
        and eligible_easy_fraction >= minimum_easy
        and median_spread is not None
        and median_spread <= maximum_spread
        and account_ready
        and (not require_locates or locate_coverage >= 0.95)
    )
    breakthrough = bool(statistical.get("statistical_gate_passed", False) and operational)
    return {
        **statistical,
        "execution_coverage": execution_coverage,
        "integrity_gate_passed": integrity_passed,
        "duplicate_observations": duplicate_observations,
        "missing_eligibility_decisions": int(len(ledger_ids - eligibility_ids)),
        "unknown_eligibility_decisions": int(len(unknown_eligibility_ids)),
        "rejected_signal_executions": int(len(rejected_execution_ids)),
        "eligibility_coverage": eligibility_coverage,
        "broker_eligible_signals": int(len(eligible_ids)),
        "broker_rejected_signals": int(len(eligibility) - len(eligible_ids)),
        "minimum_execution_coverage": minimum_execution,
        "broker_snapshot_coverage": snapshot_coverage,
        "minimum_snapshot_coverage": minimum_snapshot,
        "shortable_fraction": shortable_fraction,
        "easy_to_borrow_fraction": easy_fraction,
        "eligible_shortable_fraction": eligible_shortable_fraction,
        "eligible_easy_to_borrow_fraction": eligible_easy_fraction,
        "median_touch_spread_bps": median_spread,
        "maximum_median_spread_bps": maximum_spread,
        "actual_locate_coverage": locate_coverage,
        "actual_locates_required_count": int(len(required_locate_ids)),
        "actual_locates_confirmed_count": int(len(required_locate_ids & confirmed_locate_ids)),
        "unknown_locate_decisions": int(len(unknown_locate_ids)),
        "actual_locates_required": require_locates,
        "account_controls_ready": account_ready,
        "operational_gate_passed": operational,
        "breakthrough": breakthrough,
        "verdict": "BREAKTHROUGH" if breakthrough else "CONTINUE_FORWARD_COLLECTION",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine locked forward statistical and operational evidence.")
    parser.add_argument("--config", default="config/alpha_factory.yaml")
    parser.add_argument("--assessment", default="reports/forward_observation/assessment.json")
    parser.add_argument("--ledger", default="reports/forward_observation/ledger.csv")
    parser.add_argument("--snapshots", default="reports/forward_observation/operational_snapshots.csv")
    parser.add_argument("--executions", default="reports/forward_observation/execution_evaluation.csv")
    parser.add_argument("--accounts", default="reports/forward_observation/account_snapshots.csv")
    parser.add_argument("--eligibility", default="reports/forward_observation/eligibility.csv")
    parser.add_argument("--locates", default="reports/forward_observation/locate_evidence.csv")
    parser.add_argument("--output", default="reports/forward_observation/verdict.json")
    args = parser.parse_args()

    statistical_path = Path(args.assessment)
    statistical = json.loads(statistical_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(args.config).read_text())
    observation_config = config.get("forward_observation", {})
    gates = observation_config.get("operational_gates", {})
    ledger = read_csv(Path(args.ledger))
    eligibility = read_csv(Path(args.eligibility))
    if len(ledger) and len(eligibility):
        from src.forward_quote_capture import observation_id
        eligible_ids = set(
            eligibility.loc[eligibility["broker_eligible"].fillna(False).astype(bool), "observation_id"].astype(str)
        )
        executable_ledger = ledger.loc[
            ledger.apply(lambda row: observation_id(row) in eligible_ids, axis=1)
        ].copy()
        statistical = evaluate_evidence(executable_ledger, observation_config.get("gates", {}))
    result = combined_verdict(
        statistical, ledger, read_csv(Path(args.snapshots)), read_csv(Path(args.executions)), gates,
        read_csv(Path(args.accounts)), eligibility,
        read_csv(Path(args.locates)),
    )
    output = Path(args.output)
    atomic_write_json(result, output)
    print(f"Forward verdict: {result['verdict']}. Orders remain disabled.")


if __name__ == "__main__":
    main()
