from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def read_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def scalar(frame: pd.DataFrame, column: str, default: float = np.nan) -> float:
    if frame.empty or column not in frame.columns:
        return default
    value = frame.iloc[0][column]
    return float(value) if pd.notna(value) else default


def decision_status(locked_survivors: int, cagr: float, drawdown: float, target: float, drawdown_limit: float) -> tuple[str, str]:
    if locked_survivors == 0:
        return "REJECTED", "No hypothesis survived the locked out-of-sample test. Expand the strategy families; do not optimize these rules further."
    if not np.isfinite(cagr):
        return "INCOMPLETE", "Statistical survivors exist, but the portfolio could not produce a valid CAGR. Inspect trade generation and capacity rejections."
    if cagr < target:
        return "RESEARCH EDGE, TARGET MISSED", "Keep the robust survivors as research assets, but add independent strategy families before considering deployment."
    if drawdown > drawdown_limit:
        return "RETURN TARGET MET, RISK FAILED", "The backtest reached the return target only with excessive drawdown. Reduce concentration and stress position sizing."
    return "PAPER-TRADING CANDIDATE", "The historical gates passed. This is not production approval; proceed to walk-forward paper trading and execution monitoring."


def fmt_pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:.2%}"


def build_report(output_dir: Path, config: dict) -> str:
    validation = read_csv_or_empty(output_dir / "alpha_factory_oos_validation.csv")
    survivors = read_csv_or_empty(output_dir / "alpha_factory_locked_test_survivors.csv")
    portfolio = read_csv_or_empty(output_dir / "alpha_portfolio_summary.csv")
    selected = read_csv_or_empty(output_dir / "alpha_portfolio_selected_candidates.csv")
    trades = read_csv_or_empty(output_dir / "alpha_portfolio_trades.csv")
    rejected = read_csv_or_empty(output_dir / "alpha_portfolio_rejections.csv")

    train_discoveries = int(validation.get("train_discovery_pass", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    validation_survivors = int(validation.get("validation_pass", pd.Series(dtype=bool)).fillna(False).astype(bool).sum())
    locked_survivors = len(survivors)
    cagr = scalar(portfolio, "cagr")
    drawdown = scalar(portfolio, "max_drawdown")
    sharpe = scalar(portfolio, "sharpe")
    win_rate = scalar(portfolio, "win_rate")
    target = float(config.get("alpha_portfolio", {}).get("target_cagr", 0.40))
    drawdown_limit = float(config.get("alpha_portfolio", {}).get("maximum_acceptable_drawdown", 0.25))
    status, action = decision_status(locked_survivors, cagr, drawdown, target, drawdown_limit)

    family_lines = []
    if not survivors.empty and "family" in survivors:
        counts = survivors.groupby(["family", "direction"]).size().sort_values(ascending=False)
        family_lines = [f"- {family} / {direction}: {int(count)}" for (family, direction), count in counts.items()]
    if not family_lines:
        family_lines = ["- None"]

    return "\n".join([
        "# Alpha Factory Research Decision",
        "",
        f"**Status: {status}**",
        "",
        "## Validation funnel",
        "",
        f"- Candidates evaluated: {len(validation)}",
        f"- Train discoveries after FDR: {train_discoveries}",
        f"- Validation survivors: {validation_survivors}",
        f"- Locked-test survivors: {locked_survivors}",
        f"- Portfolio representatives selected: {len(selected)}",
        "",
        "## Locked-test portfolio",
        "",
        f"- Completed trades: {len(trades)}",
        f"- Capacity rejections: {len(rejected)}",
        f"- CAGR: {fmt_pct(cagr)}",
        f"- Target CAGR: {fmt_pct(target)}",
        f"- Maximum drawdown: {fmt_pct(drawdown)}",
        f"- Drawdown limit: {fmt_pct(drawdown_limit)}",
        f"- Sharpe ratio: {'n/a' if not np.isfinite(sharpe) else f'{sharpe:.2f}'}",
        f"- Win rate: {fmt_pct(win_rate)}",
        "",
        "## Surviving strategy mix",
        "",
        *family_lines,
        "",
        "## Decision",
        "",
        action,
        "",
        "## Non-negotiable interpretation",
        "",
        "A 40% historical CAGR is not proof that 40% can be earned live. Production approval remains false until the strategy survives paper trading, execution slippage checks, borrow/locate stress for shorts, and a forward drawdown period.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/alpha_factory.yaml")
    parser.add_argument("--prices", default="data/processed/daily_prices.parquet", help="Accepted for pipeline compatibility; not read by this report.")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(Path(args.config).read_text())
    report = build_report(output, config)
    path = output / "alpha_research_decision.md"
    path.write_text(report, encoding="utf-8")
    print(report)
    print(f"Decision report written to {path}")


if __name__ == "__main__":
    main()
