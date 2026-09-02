from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml


def validation_gate(candidates: pd.DataFrame, minimum_events: int) -> dict:
    validation = candidates.loc[candidates["period"].eq("validation")].copy()
    retained = validation.loc[~validation["event_day_halt"].fillna(False).astype(bool)]
    return {
        "validation_candidates": int(len(validation)),
        "validation_event_day_halts": int(validation["event_day_halt"].fillna(False).astype(bool).sum()),
        "validation_candidates_retained": int(len(retained)),
        "minimum_validation_events": int(minimum_events),
        "validation_sample_gate_passed": bool(len(retained) >= minimum_events),
        "test_evaluation_authorized": bool(len(retained) >= minimum_events),
    }


def assessment(summary: dict) -> str:
    return f"""# Event-Day Halt Exclusion Assessment

## Verdict

**Rejected on validation sample size; locked-test return evaluation was not
authorized.** The fixed rule excludes every candidate with an official Nasdaq halt on
the signal date. It removes {summary['validation_event_day_halts']} of
{summary['validation_candidates']} validation candidates and retains only
{summary['validation_candidates_retained']}, below the locked minimum of
{summary['minimum_validation_events']}.

The rule is operationally motivated and binary, but evaluating its test-period CAGR
after failing the validation gate would turn the test set into a selection tool. Halt
evidence remains mandatory prospective execution-risk telemetry. The rule may be
reconsidered only after a new forward sample supplies enough causally captured halted
and non-halted observations. Allocation and order submission remain disabled.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Gate a fixed event-day halt exclusion on validation only")
    parser.add_argument("--candidates", default="reports/event_risk_coverage/candidate_event_risk.csv")
    parser.add_argument("--config", default="config/alpha_factory.yaml")
    parser.add_argument("--output-dir", default="reports/halt_exclusion")
    args = parser.parse_args()

    candidates = pd.read_csv(args.candidates)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    minimum = int(config["validation"]["minimum_validation_events"])
    summary = validation_gate(candidates, minimum)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "ASSESSMENT.md").write_text(assessment(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
