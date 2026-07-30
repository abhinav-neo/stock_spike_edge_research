from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run(cmd: list[str], log_path: Path) -> dict:
    print(f"\n>>> {' '.join(cmd)}")
    completed = subprocess.run(cmd, text=True, capture_output=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"COMMAND\n{' '.join(cmd)}\n\nSTDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}",
        encoding="utf-8",
    )
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    return {
        "command": cmd,
        "return_code": completed.returncode,
        "log": str(log_path),
        "status": "completed" if completed.returncode == 0 else "failed",
    }


def read_columns(path: Path) -> list[str]:
    if path.suffix.lower() == ".parquet":
        return list(pd.read_parquet(path).columns)
    return list(pd.read_csv(path, nrows=1).columns)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete V5 robustness suite with one command")
    parser.add_argument("--predictions", default="reports/v5_random_forest/predictions.csv")
    parser.add_argument("--features", default="data/processed/events_features_v5.parquet")
    parser.add_argument("--output-dir", default="reports/v5_validation_suite")
    parser.add_argument("--target", default="forward_return_5d")
    parser.add_argument("--first-test-year", type=int, default=2020)
    parser.add_argument("--min-price", type=float, default=5.0)
    parser.add_argument("--min-dollar-volume", type=float, default=5_000_000.0)
    parser.add_argument("--min-market-cap", type=float, default=None)
    args = parser.parse_args()

    predictions = Path(args.predictions)
    features = Path(args.features)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if not predictions.exists():
        raise FileNotFoundError(f"Predictions file not found: {predictions}")
    if not features.exists():
        raise FileNotFoundError(f"Feature file not found: {features}")

    prediction_columns = read_columns(predictions)
    feature_columns = read_columns(features)
    available = set(prediction_columns) | set(feature_columns)

    price_columns = ["entry_price", "price", "close"]
    dollar_volume_columns = ["avg_dollar_volume_20d", "dollar_volume_20d", "dollar_volume"]
    market_cap_columns = ["market_cap"]

    diagnostics = {
        "predictions": str(predictions),
        "features": str(features),
        "prediction_columns": prediction_columns,
        "feature_columns": feature_columns,
        "price_filter_available": any(c in available for c in price_columns),
        "dollar_volume_filter_available": any(c in available for c in dollar_volume_columns),
        "market_cap_filter_available": any(c in available for c in market_cap_columns),
    }

    runs: dict[str, dict] = {}
    python = sys.executable

    common = [
        python, "-m", "src.ranked_portfolio_backtest",
        "--input", str(predictions),
        "--features-input", str(features),
        "--target", args.target,
        "--period", "test",
        "--threshold-period", "validation",
        "--side", "short",
        "--fraction", "0.10",
        "--holding-days", "5",
        "--initial-capital", "50000",
        "--max-positions", "5",
        "--cost-bps", "100",
        "--borrow-bps-per-day", "50",
    ]

    overlap_dir = output / "short_overlap_fixed"
    runs["overlap_fixed"] = run(
        common + ["--output-dir", str(overlap_dir)],
        output / "logs" / "overlap_fixed.log",
    )

    tradeable_missing: list[str] = []
    tradeable_cmd = common.copy()
    if diagnostics["price_filter_available"]:
        tradeable_cmd += ["--min-price", str(args.min_price)]
    else:
        tradeable_missing.append("price")
    if diagnostics["dollar_volume_filter_available"]:
        tradeable_cmd += ["--min-dollar-volume", str(args.min_dollar_volume)]
    else:
        tradeable_missing.append("dollar_volume")
    if args.min_market_cap is not None:
        if diagnostics["market_cap_filter_available"]:
            tradeable_cmd += ["--min-market-cap", str(args.min_market_cap)]
        else:
            tradeable_missing.append("market_cap")

    tradeable_dir = output / "short_tradeable"
    if len(tradeable_missing) < 2:
        runs["tradeable"] = run(
            tradeable_cmd + ["--output-dir", str(tradeable_dir)],
            output / "logs" / "tradeable.log",
        )
        runs["tradeable"]["filters_skipped"] = tradeable_missing
    else:
        runs["tradeable"] = {
            "status": "skipped",
            "reason": f"Required fields unavailable: {', '.join(tradeable_missing)}",
        }

    walk_dir = output / "walk_forward"
    runs["walk_forward"] = run(
        [
            python, "-m", "src.walk_forward_validation",
            "--input", str(features),
            "--target", args.target,
            "--model", "random_forest",
            "--first-test-year", str(args.first_test_year),
            "--selection-fraction", "0.10",
            "--output-dir", str(walk_dir),
        ],
        output / "logs" / "walk_forward.log",
    )

    results = {
        "diagnostics": diagnostics,
        "runs": runs,
        "overlap_summary": load_json(overlap_dir / "portfolio_summary.json"),
        "tradeable_summary": load_json(tradeable_dir / "portfolio_summary.json"),
        "walk_forward_summary": load_json(walk_dir / "walk_forward_summary.json"),
    }
    (output / "suite_summary.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    lines = [
        "# V5 Validation Suite Report",
        "",
        "## Run status",
    ]
    for name, result in runs.items():
        lines.append(f"- **{name}**: {result.get('status')}" + (f" — {result.get('reason')}" if result.get("reason") else ""))

    for title, key in [
        ("Overlap-protected short backtest", "overlap_summary"),
        ("Tradeability-filtered short backtest", "tradeable_summary"),
        ("Walk-forward validation", "walk_forward_summary"),
    ]:
        lines += ["", f"## {title}", "", "```json", json.dumps(results[key], indent=2, default=str), "```"]

    lines += [
        "",
        "## Data availability",
        f"- Price field available: {diagnostics['price_filter_available']}",
        f"- Dollar-volume field available: {diagnostics['dollar_volume_filter_available']}",
        f"- Market-cap field available: {diagnostics['market_cap_filter_available']}",
    ]
    (output / "REPORT_FOR_REVIEW.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nSuite complete. Review: {output / 'REPORT_FOR_REVIEW.md'}")
    print(f"Machine-readable summary: {output / 'suite_summary.json'}")


if __name__ == "__main__":
    main()
