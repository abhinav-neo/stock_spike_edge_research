from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date


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
    result.append([sys.executable, "-B", "-m", "src.alpaca_operational_snapshot", "--date", end_date])
    result.append([sys.executable, "-B", "-m", "src.forward_eligibility"])
    result.append([sys.executable, "-B", "-m", "src.alpaca_account_snapshot", "--date", end_date])
    result.append([sys.executable, "-B", "-m", "src.forward_quote_capture"])
    result.append([sys.executable, "-B", "-m", "src.forward_execution_evaluation"])
    result.append([sys.executable, "-B", "-m", "src.forward_breakthrough_assessment"])
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
