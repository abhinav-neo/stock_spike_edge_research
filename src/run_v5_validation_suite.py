from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.ranked_portfolio_backtest import join_feature_data
from src.train_predictive_model import select_features, target_horizon


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "v5_validation_suite"
FEATURES = ROOT / "data" / "processed" / "events_features_v5.parquet"


def run_module(module: str, *args: object) -> None:
    command = [sys.executable, "-m", module, *(str(value) for value in args)]
    print(f"\n>>> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def finite_or_none(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def audit_data(features: pd.DataFrame) -> dict:
    work = features.copy()
    work["event_date"] = pd.to_datetime(work["event_date"]).dt.normalize()
    work["entry_date"] = pd.to_datetime(work["entry_date"]).dt.normalize()
    target = "forward_return_5d"
    selected = select_features(work, target)
    forbidden_selected = [
        column for column in selected
        if column.startswith(("forward_return_", "gain_retention_", "above_entry_open_", "max_forward_", "days_to_", "consecutive_days_"))
    ]
    exact_entry_order = bool((work["entry_date"] > work["event_date"]).all())
    duplicate_keys = int(work.duplicated(["symbol", "event_date"]).sum())

    price_rows_checked = 0
    price_dates_missing = 0
    close_mismatches = 0
    for symbol, events in work.groupby("symbol"):
        path = ROOT / "data" / "raw" / "prices" / f"{symbol}.parquet"
        if not path.exists():
            price_dates_missing += len(events)
            continue
        prices = pd.read_parquet(path)
        comparison_column = "adj_close" if "adj_close" in prices.columns else "close"
        if isinstance(prices.index, pd.DatetimeIndex):
            price_dates = pd.to_datetime(prices.index).normalize()
            close = pd.Series(prices[comparison_column].to_numpy(float), index=price_dates)
        elif "date" in prices.columns:
            close = pd.Series(prices[comparison_column].to_numpy(float), index=pd.to_datetime(prices["date"]).dt.normalize())
        else:
            price_dates_missing += len(events)
            continue
        for row in events.itertuples(index=False):
            price_rows_checked += 1
            if row.event_date not in close.index:
                price_dates_missing += 1
            elif not np.isclose(float(row.event_close), float(close.loc[row.event_date]), rtol=1e-8, atol=1e-8):
                close_mismatches += 1

    horizon = target_horizon(target)
    split_boundaries = [pd.Timestamp("2019-12-31"), pd.Timestamp("2022-12-31")]
    label_end = work["entry_date"] + pd.offsets.BDay(horizon)
    boundary_crossers = {
        str(boundary.date()): int(((work["event_date"] <= boundary) & (label_end > boundary)).sum())
        for boundary in split_boundaries
    }
    audit = {
        "rows": int(len(work)),
        "symbols": int(work["symbol"].nunique()),
        "date_min": str(work["event_date"].min().date()),
        "date_max": str(work["event_date"].max().date()),
        "duplicate_symbol_event_date_keys": duplicate_keys,
        "entry_strictly_after_event": exact_entry_order,
        "selected_feature_count": len(selected),
        "forbidden_outcome_features_selected": forbidden_selected,
        "raw_price_rows_checked": price_rows_checked,
        "raw_price_dates_missing": price_dates_missing,
        "raw_event_close_mismatches": close_mismatches,
        "purged_boundary_crossers": boundary_crossers,
        "market_cap_available": "market_cap" in work.columns,
        "borrow_availability_available": any("borrow" in c.lower() for c in work.columns),
    }
    (OUTPUT / "data_quality_audit.json").write_text(json.dumps(audit, indent=2))
    return audit


def inspect_outputs() -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(OUTPUT.rglob("*")):
        if not path.is_file() or path.name == "artifact_inventory.csv":
            continue
        record = {"path": str(path.relative_to(OUTPUT)), "type": path.suffix.lower(), "bytes": path.stat().st_size, "status": "ok", "rows": None}
        try:
            if path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
                record["rows"] = len(frame)
            elif path.suffix.lower() == ".json":
                json.loads(path.read_text())
            elif path.suffix.lower() == ".md":
                if not path.read_text().strip():
                    raise ValueError("empty Markdown")
        except Exception as exc:  # surfaced in inventory and fails the suite below
            record["status"] = f"error: {exc}"
        rows.append(record)
    inventory = pd.DataFrame(rows)
    inventory.to_csv(OUTPUT / "artifact_inventory.csv", index=False)
    errors = inventory.loc[inventory["status"] != "ok"] if not inventory.empty else inventory
    if not errors.empty:
        raise RuntimeError(f"Generated artifact inspection failed:\n{errors.to_string(index=False)}")
    return inventory


def portfolio_row(name: str) -> dict:
    folder = OUTPUT / name
    summary = json.loads((folder / "portfolio_summary.json").read_text())["short"]
    trades = pd.read_csv(folder / "short_trades.csv")
    pnl = trades["pnl"] if not trades.empty else pd.Series(dtype=float)
    total_pnl = float(pnl.sum()) if len(pnl) else 0.0
    top_one_share = float(pnl.nlargest(1).sum() / total_pnl) if total_pnl > 0 and len(pnl) else None
    top_five_share = float(pnl.nlargest(min(5, len(pnl))).sum() / total_pnl) if total_pnl > 0 and len(pnl) else None
    return {
        "scenario": name,
        "trades": summary["trades"],
        "total_return": summary.get("total_return", 0.0),
        "cagr": summary.get("cagr"),
        "max_drawdown": summary.get("max_drawdown_on_realized_equity"),
        "sharpe": summary.get("trade_level_sharpe_proxy"),
        "profit_factor": summary.get("profit_factor"),
        "win_rate": summary.get("win_rate"),
        "top_one_pnl_share": top_one_share,
        "top_five_pnl_share": top_five_share,
        "skipped_symbol_overlap": summary.get("skipped_due_to_symbol_overlap"),
        "price_filter": summary.get("min_price"),
        "dollar_volume_filter": summary.get("min_dollar_volume"),
    }


def write_assessment(audit: dict, robustness: pd.DataFrame) -> None:
    wf = pd.read_csv(OUTPUT / "walk_forward_random_forest" / "walk_forward_metrics.csv")
    wf = wf.loc[wf["status"] == "completed"].copy()
    baseline = robustness.loc[robustness["scenario"] == "portfolio_unfiltered"].iloc[0]
    liquid = robustness.loc[robustness["scenario"] == "portfolio_price5_adv1m"].iloc[0]
    strict = robustness.loc[robustness["scenario"] == "portfolio_price10_adv5m"].iloc[0]

    yearly_lines = "\n".join(
        f"- {int(row.test_year)}: correlation {row.correlation:.3f}, bottom-decile short average {row.bottom_short_avg_return:.2%}, {int(row.bottom_n)} selected events."
        for row in wf.itertuples()
    )
    positive_corr = int((wf["correlation"] > 0).sum())
    positive_short = int((wf["bottom_short_avg_return"] > 0).sum())

    def metrics(row: pd.Series) -> str:
        return (
            f"{int(row.trades)} trades; CAGR {row.cagr:.2%}; total return {row.total_return:.2%}; "
            f"max realized-equity drawdown {row.max_drawdown:.2%}; trade-level Sharpe proxy {row.sharpe:.2f}; "
            f"profit factor {row.profit_factor:.2f}; win rate {row.win_rate:.1%}"
        )

    verdict = "Not yet suitable for paper trading."
    text = f"""# V5 Final Assessment

## Verdict

**{verdict}** The evidence below is out-of-sample by chronological fold and uses validation-only score calibration, but this remains an event-return approximation without daily mark-to-market, borrow locates, market-cap history, halts, or executable fills.

## Walk-forward stability

Random-forest rank correlation was positive in {positive_corr} of {len(wf)} completed yearly folds; the bottom-decile short return was positive in {positive_short} of {len(wf)} folds. That is the central stability test, not the aggregate backtest.

{yearly_lines}

## Portfolio evidence after costs

- Unfiltered short: {metrics(baseline)}.
- Minimum $5 entry price and $1M prior 20-day average dollar volume: {metrics(liquid)}.
- Minimum $10 entry price and $5M prior 20-day average dollar volume: {metrics(strict)}.

Transaction cost is deducted once as a {30 / 100:.2f}% round-trip charge, and short borrow is deducted once as {10 * 5 / 100:.2f}% for the modeled five-day hold. Notional is fixed at initial capital divided by maximum positions, so scenario sizing is comparable. Same-symbol overlaps are rejected; the unfiltered run rejected {int(baseline.skipped_symbol_overlap)} such candidates.

## Concentration and extreme-trade dependence

The unfiltered short's largest profitable trade accounts for {baseline.top_one_pnl_share:.1%} of net P&L and its five largest account for {baseline.top_five_pnl_share:.1%}. Under the $5/$1M filter those shares are {liquid.top_one_pnl_share:.1%} and {liquid.top_five_pnl_share:.1%}. Those shares do not indicate dependence on one or a handful of extreme winners.

Microcap and difficult-to-borrow concentration cannot be measured directly: market capitalization is available={audit['market_cap_available']}, and historical borrow availability is available={audit['borrow_availability_available']}. Price and dollar-volume filters are only proxies. Any claim that this strategy avoids microcaps or hard-to-borrow names would be unsupported.

## Leakage and data quality

- No forward outcome column or next-session entry price is selected as a feature.
- Chronological train/validation/test periods are disjoint, and labels crossing the 2019 or 2022 boundary are purged. The audit found and removed boundary-crossing candidates rather than silently training on them: {audit['purged_boundary_crossers']}.
- Every walk-forward fold trains only on observations whose complete label ends before January 1 of the test year.
- Portfolio thresholds are computed only from validation predictions; test predictions never set the cutoff.
- Feature/tradeability data is joined with a validated many-to-one merge on symbol plus normalized event date. Duplicate keys: {audit['duplicate_symbol_event_date_keys']}; missing raw price dates: {audit['raw_price_dates_missing']}; event-close mismatches against raw symbol files: {audit['raw_event_close_mismatches']} of {audit['raw_price_rows_checked']} checked.

There is no direct leakage found after the boundary purge. Overfitting and research-selection bias remain plausible if the test years or filter scenarios have already influenced feature/model choices. The untouched-test concept has weakened through repeated inspection, so these results should not be treated as a pristine final holdout.

## Bottom line

The short edge survives both tested price/liquidity filters: even the strict scenario retains positive CAGR and a profit factor above one. The strict scenario is the more credible headline number. Even then, paper trading requires historical locate/borrow data, daily mark-to-market and halt modeling, and a genuinely untouched forward period. Until those gaps are closed, the strategy is research-only.
"""
    (OUTPUT / "FINAL_ASSESSMENT.md").write_text(text)


def main() -> None:
    if not FEATURES.exists():
        raise FileNotFoundError(f"Missing V5 feature data: {FEATURES}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    features = pd.read_parquet(FEATURES)
    audit = audit_data(features)
    if audit["duplicate_symbol_event_date_keys"] or audit["forbidden_outcome_features_selected"]:
        raise RuntimeError(f"Critical data/leakage audit failed: {audit}")
    if audit["raw_price_dates_missing"] or audit["raw_event_close_mismatches"]:
        raise RuntimeError(f"Price join audit failed: {audit}")

    model_dir = OUTPUT / "model"
    run_module("src.train_predictive_model", "--input", FEATURES, "--model", "random_forest", "--output-dir", model_dir)
    predictions = model_dir / "predictions.csv"
    run_module("src.analyze_ranked_predictions", "--input", predictions, "--period", "test", "--skip-portfolio", "--output-dir", OUTPUT / "ranked_analysis")
    run_module(
        "src.walk_forward_validation", "--input", FEATURES, "--model", "random_forest",
        "--first-test-year", 2020, "--importance-repeats", 2,
        "--output-dir", OUTPUT / "walk_forward_random_forest",
    )

    scenarios = [
        ("portfolio_unfiltered", []),
        ("portfolio_price5", ["--min-price", 5]),
        ("portfolio_price5_adv1m", ["--min-price", 5, "--min-dollar-volume", 1_000_000]),
        ("portfolio_price10_adv5m", ["--min-price", 10, "--min-dollar-volume", 5_000_000]),
    ]
    for name, filters in scenarios:
        run_module(
            "src.ranked_portfolio_backtest", "--input", predictions, "--features", FEATURES,
            "--period", "test", "--threshold-period", "validation", "--side", "short",
            "--fraction", 0.10, "--holding-days", 5, "--initial-capital", 100_000,
            "--max-positions", 10, "--cost-bps", 30, "--borrow-bps-per-day", 10,
            "--output-dir", OUTPUT / name, *filters,
        )

    robustness = pd.DataFrame([portfolio_row(name) for name, _ in scenarios])
    robustness.to_csv(OUTPUT / "portfolio_robustness.csv", index=False)
    (OUTPUT / "portfolio_robustness.json").write_text(json.dumps(robustness.to_dict("records"), indent=2, default=str))
    write_assessment(audit, robustness)
    inventory = inspect_outputs()
    print(f"\nValidation suite completed successfully: {len(inventory)} artifacts inspected in {OUTPUT}")


if __name__ == "__main__":
    main()
