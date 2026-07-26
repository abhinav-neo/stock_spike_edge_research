from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_stage(label: str, module: str, arguments: list[str]) -> None:
    command = [sys.executable, "-B", "-m", module, *arguments]
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

    common = ["--config", args.config, "--prices", args.prices, "--output-dir", args.output_dir]
    stages: list[tuple[str, str, list[str]]] = []
    if not args.skip_discovery:
        stages.append(("candidate discovery", "src.alpha_factory", common))
    stages.extend([
        ("out-of-sample validation", "src.validate_alpha_factory", common),
        (
            "portfolio simulation",
            "src.alpha_portfolio",
            [*common, "--survivors", str(Path(args.output_dir) / "alpha_factory_locked_test_survivors.csv")],
        ),
        ("decision report", "src.alpha_research_report", common),
    ])

    for label, module, arguments in stages:
        run_stage(label, module, arguments)

    print(f"\nPipeline complete. Open {Path(args.output_dir) / 'alpha_research_decision.md'}")


if __name__ == "__main__":
    main()
