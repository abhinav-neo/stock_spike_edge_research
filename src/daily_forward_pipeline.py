from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date
from pathlib import Path

import yaml


def execution_protocol(config_path: str = "config/alpha_factory.yaml") -> dict[str, str]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return config["forward_observation"]["execution_protocol"]


def commands(end_date: str, skip_data_update: bool, symbols: list[str]) -> list[list[str]]:
    result: list[list[str]] = []
    if not skip_data_update:
        update = [
            sys.executable, "-B", "-m", "src.daily_market_data_updater", "--end-date", end_date,
            "--full-existing-universe",
        ]
        if symbols:
            update.extend(["--symbols", *symbols])
        result.append(update)
    result.append([sys.executable, "-B", "-m", "src.forward_observation"])
    result.append([sys.executable, "-B", "-m", "src.forward_event_risk"])
    result.append([sys.executable, "-B", "-m", "src.alpaca_operational_snapshot", "--date", end_date])
    result.append([sys.executable, "-B", "-m", "src.forward_eligibility"])
    result.append([sys.executable, "-B", "-m", "src.alpaca_locate_evidence"])
    result.append([sys.executable, "-B", "-m", "src.alpaca_account_snapshot", "--date", end_date])
    protocol = execution_protocol()
    protocol_start = protocol["effective_entry_date"]
    quote_root = protocol["quote_root"]
    execution_output = protocol["execution_output"]
    result.append([
        sys.executable, "-B", "-m", "src.forward_quote_capture",
        "--feed", protocol["feed"], "--minimum-entry-date", protocol_start,
        "--output", quote_root, "--summary", protocol["quote_summary"],
    ])
    result.append([
        sys.executable, "-B", "-m", "src.forward_execution_evaluation",
        "--minimum-entry-date", protocol_start, "--quotes", quote_root,
        "--output", execution_output,
    ])
    result.append([
        sys.executable, "-B", "-m", "src.forward_breakthrough_assessment",
        "--executions", execution_output,
    ])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Update data and record zero-capital locked forward observations.")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--skip-data-update", action="store_true")
    parser.add_argument("--symbols", nargs="*", default=[])
    args = parser.parse_args()

    for command in commands(args.end_date, args.skip_data_update, args.symbols):
        subprocess.run(command, check=True)
    print("Forward observation completed. Allocation and order submission remain disabled.")


if __name__ == "__main__":
    main()
