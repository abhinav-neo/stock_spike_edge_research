from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import date


def run_step(module: str, extra_args: list[str] | None = None) -> None:
    command = [sys.executable, "-B", "-m", module, *(extra_args or [])]
    print(f"\n=== Running {module} ===")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the safe daily paper-trading workflow.")
    parser.add_argument("--skip-data-update", action="store_true")
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument("--symbols", nargs="*", default=[])
    args = parser.parse_args()

    if not args.skip_data_update:
        updater_args = ["--end-date", args.end_date]
        if args.symbols:
            updater_args.extend(["--symbols", *args.symbols])
        run_step("src.daily_market_data_updater", updater_args)

    run_step("src.paper_fill_tracker")
    run_step("src.paper_position_lifecycle")
    print("\nDaily paper pipeline completed. Live order submission remains disabled.")


if __name__ == "__main__":
    main()
