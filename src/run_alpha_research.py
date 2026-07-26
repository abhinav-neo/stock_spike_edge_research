from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


STAGES = [
    ("candidate discovery", "src.alpha_factory"),
    ("out-of-sample validation", "src.validate_alpha_factory"),
    ("portfolio simulation", "src.alpha_portfolio"),
    ("decision report", "src.alpha_research_report"),
]


def run_stage(label: str, module: str, common_args: list[str]) -> None:
    command = [sys.executable, "-B", "-m", module, *common_args]
    print(f"\n=== {label.upper()} ===")
    print(" ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"Stage failed: {label} (exit code {completed.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete Alpha Factory research pipeline.")
    parser.add_argument("--config", default="config/alpha_factory.yaml")
    parser.add_argument("--prices", default="data/processed/daily_prices.parquet")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--skip-discovery", action="store_true")
    args = parser.parse_args()

    if not Path(args.config).exists():
        raise FileNotFoundError(f"Missing config: {args.config}")
    if not Path(args.prices).exists():
        raise FileNotFoundError(f"Missing price data: {args.prices}")

    common_args = ["--config", args.config, "--prices", args.prices, "--output-dir", args.output_dir]
    stages = STAGES[1:] if args.skip_discovery else STAGES
    for label, module in stages:
        run_stage(label, module, common_args)

    print(f"\nPipeline complete. Open {Path(args.output_dir) / 'alpha_research_decision.md'}")


if __name__ == "__main__":
    main()
