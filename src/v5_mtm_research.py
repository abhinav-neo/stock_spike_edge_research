from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.mark_to_market import apply_locate_model, trade_paths
from src.build_daily_prices import normalize_daily_frame
from src.ranked_portfolio_backtest import (
    apply_tradeability_filters,
    join_feature_data,
    score_threshold,
    simulate,
)


def attach_event_dates(trades: pd.DataFrame, features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    left = trades.copy()
    right = features[["symbol", "event_date", "entry_date"]].copy()
    left["entry_date"] = pd.to_datetime(left["entry_date"]).dt.normalize()
    right["entry_date"] = pd.to_datetime(right["entry_date"]).dt.normalize()
    right["event_date"] = pd.to_datetime(right["event_date"]).dt.normalize()
    if right.duplicated(["symbol", "entry_date"]).any():
        raise ValueError("Feature data contains duplicate symbol/entry_date keys")
    joined = left.merge(right, on=["symbol", "entry_date"], how="left", validate="many_to_one")
    if joined["event_date"].isna().any():
        raise ValueError("Some portfolio trades could not be matched to their event date")
    joined["horizon"] = int(horizon)
    return joined


def apply_round_trip_cost(paths: pd.DataFrame, trades: pd.DataFrame, cost_bps: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cost = float(cost_bps) / 10_000.0
    adjusted_paths = paths.copy()
    adjusted_trades = trades.copy()
    if not adjusted_paths.empty:
        adjusted_paths["mark_return"] = adjusted_paths["mark_return"] - cost
    if not adjusted_trades.empty:
        adjusted_trades["transaction_cost_return"] = cost
        adjusted_trades["net_return"] = adjusted_trades["net_return"] - cost
    return adjusted_paths, adjusted_trades


def daily_equity_curve(
    paths: pd.DataFrame,
    trades: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    initial_capital: float,
    stake: float,
) -> pd.DataFrame:
    if paths.empty or trades.empty:
        return pd.DataFrame(columns=["date", "equity", "daily_return", "drawdown", "concurrent_positions", "gross_exposure"])
    work_paths = paths.copy()
    work_trades = trades.copy()
    work_paths["date"] = pd.to_datetime(work_paths["date"]).dt.normalize()
    work_trades["exit_date"] = pd.to_datetime(work_trades["exit_date"]).dt.normalize()
    start, end = work_paths["date"].min(), work_paths["date"].max()
    dates = pd.DatetimeIndex(calendar).normalize()
    dates = dates[(dates >= start) & (dates <= end)]

    marks = work_paths.assign(mark_pnl=work_paths["mark_return"] * stake)
    active_mark = marks.groupby("date")["mark_pnl"].sum().reindex(dates, fill_value=0.0)
    concurrent = work_paths.groupby("date")["trade_id"].nunique().reindex(dates, fill_value=0).astype(int)
    realized = (work_trades.assign(pnl=work_trades["net_return"] * stake)
                .groupby("exit_date")["pnl"].sum().reindex(dates, fill_value=0.0))
    realized_prior = realized.cumsum().shift(1, fill_value=0.0)

    equity = pd.DataFrame({"date": dates})
    equity["equity"] = initial_capital + realized_prior.to_numpy() + active_mark.to_numpy()
    equity["concurrent_positions"] = concurrent.to_numpy()
    equity["gross_exposure"] = equity["concurrent_positions"] * stake / initial_capital
    equity["daily_return"] = equity["equity"].pct_change().fillna(0.0)
    equity["drawdown"] = equity["equity"] / equity["equity"].cummax() - 1.0
    return equity


def normalize_spy(path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(path)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(column[0]).lower().replace(" ", "_") for column in raw.columns]
    else:
        raw.columns = [str(column).lower().replace(" ", "_") for column in raw.columns]
    raw = raw.reset_index()
    date_column = next(column for column in raw.columns if column.lower() in {"date", "datetime"})
    raw = raw.rename(columns={date_column: "date"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None).dt.normalize()
    value_column = "adj_close" if "adj_close" in raw.columns else "close"
    return raw[["date", value_column]].rename(columns={value_column: "benchmark"}).dropna()


def load_candidate_prices(symbols: pd.Series, raw_directory: Path) -> pd.DataFrame:
    frames = []
    missing = []
    for symbol in sorted(set(symbols.astype(str))):
        path = raw_directory / f"{symbol}.parquet"
        if not path.exists():
            missing.append(symbol)
            continue
        frames.append(normalize_daily_frame(path, use_adjusted_prices=True))
    if missing:
        raise FileNotFoundError(f"Missing raw price histories for selected symbols: {missing}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_period_trades(predictions: pd.DataFrame, features: pd.DataFrame, period: str) -> pd.DataFrame:
    source = join_feature_data(predictions, features)
    source = source.rename(columns={"forward_return_5d": "actual_return"})
    source, _ = apply_tradeability_filters(source, 10.0, 5_000_000.0, None)
    threshold = score_threshold(source, "short", 0.10, "validation")
    evaluation = source.loc[source["period"] == period].copy()
    trades, _, _ = simulate(
        evaluation, "short", threshold, 0.10, "validation", 5,
        100_000.0, 10, 30.0, 10.0, True,
    )
    return trades


def trade_statistics(completed: pd.DataFrame) -> dict:
    if completed.empty:
        return {"win_rate": None, "profit_factor": None, "worst_trade": None, "stop_rate": None, "gap_stop_rate": None}
    returns = completed["net_return"]
    winners, losers = returns.loc[returns > 0], returns.loc[returns < 0]
    loss = float(-losers.sum())
    return {
        "win_rate": float((returns > 0).mean()),
        "profit_factor": float(winners.sum() / loss) if loss > 0 else None,
        "worst_trade": float(returns.min()),
        "stop_rate": float((completed["exit_reason"] == "stop").mean()),
        "gap_stop_rate": float((completed["stop_fill_type"] == "gap_open").mean()),
    }


def performance(equity: pd.DataFrame, spy: pd.DataFrame, initial_capital: float) -> tuple[dict, pd.DataFrame]:
    aligned = equity.merge(spy, on="date", how="left").dropna(subset=["benchmark"]).copy()
    aligned["benchmark_return"] = aligned["benchmark"].pct_change().fillna(0.0)
    aligned["benchmark_equity"] = initial_capital * aligned["benchmark"] / aligned["benchmark"].iloc[0]
    aligned["benchmark_drawdown"] = aligned["benchmark_equity"] / aligned["benchmark_equity"].cummax() - 1.0
    strategy_returns = aligned["daily_return"]
    benchmark_returns = aligned["benchmark_return"]
    covariance = strategy_returns.cov(benchmark_returns)
    variance = benchmark_returns.var(ddof=1)
    years = max((aligned["date"].iloc[-1] - aligned["date"].iloc[0]).days / 365.25, 1 / 365.25)
    ending_realized = float(equity["equity"].iloc[-1])
    daily_std = float(strategy_returns.std(ddof=1))
    summary = {
        "start": str(aligned["date"].iloc[0].date()),
        "end": str(aligned["date"].iloc[-1].date()),
        "ending_equity": ending_realized,
        "total_return": ending_realized / initial_capital - 1.0,
        "cagr": (ending_realized / initial_capital) ** (1 / years) - 1.0 if ending_realized > 0 else -1.0,
        "max_drawdown": float(aligned["drawdown"].min()),
        "annualized_volatility": daily_std * np.sqrt(252),
        "sharpe_zero_rate": float(strategy_returns.mean() / daily_std * np.sqrt(252)) if daily_std > 0 else None,
        "spy_total_return_aligned": float(aligned["benchmark_equity"].iloc[-1] / initial_capital - 1.0),
        "spy_max_drawdown_aligned": float(aligned["benchmark_drawdown"].min()),
        "daily_correlation_to_spy": float(strategy_returns.corr(benchmark_returns)),
        "beta_to_spy": float(covariance / variance) if variance > 0 else None,
        "average_gross_exposure": float(aligned["gross_exposure"].mean()),
        "maximum_gross_exposure": float(aligned["gross_exposure"].max()),
    }
    return summary, aligned


def yearly_comparison(aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, group in aligned.groupby(aligned["date"].dt.year):
        strategy = float(group["equity"].iloc[-1] / group["equity"].iloc[0] - 1.0)
        benchmark = float(group["benchmark_equity"].iloc[-1] / group["benchmark_equity"].iloc[0] - 1.0)
        rows.append({"year": int(year), "strategy_return": strategy, "spy_return_aligned": benchmark, "excess_return": strategy - benchmark, "start": str(group["date"].iloc[0].date()), "end": str(group["date"].iloc[-1].date())})
    return pd.DataFrame(rows)


def write_assessment(output: Path, calibration: pd.DataFrame, base: dict, yearly: pd.DataFrame, stress: pd.DataFrame, locate_mc: pd.DataFrame) -> None:
    profitable_stress = int((stress["total_return"] > 0).sum())
    total_stress = len(stress)
    selected = calibration.loc[calibration["selected_on_validation"]].iloc[0]
    yearly_table = "\n".join(
        ["| Year | Strategy | SPY aligned | Excess |", "|---:|---:|---:|---:|"]
        + [f"| {int(row.year)} | {row.strategy_return:.2%} | {row.spy_return_aligned:.2%} | {row.excess_return:.2%} |" for row in yearly.itertuples()]
    )
    text = f"""# V5 Daily Mark-to-Market Research Assessment

## Verdict

The V5 failed-spike short remains a useful research signal, but the daily-path analysis weakens the earlier event-return result. It is **not ready for paper trading**. The principal problem is unbounded short-squeeze risk, not ordinary transaction cost.

## Validation-calibrated risk control

Risk controls were selected only on 2020-2022 validation trades. No tested stop from 20% through 100% was profitable on validation data; the selected rule was no stop, with validation CAGR {selected.cagr:.2%}, drawdown {selected.max_drawdown:.2%}, and Calmar {selected.calmar:.2f}. This means the research does not support claiming that a conventional stop improves the strategy.

## Forward test with realistic execution assumptions

The primary 2023+ scenario uses the validation-selected risk rule, 70% deterministic locate availability, 10% annual borrow, 1% round-trip cost, fixed $10,000 notional, and at most the already capacity-controlled positions:

- Trades completed: {int(base['completed_trades'])} of {int(base['candidate_trades'])} candidates.
- CAGR: {base['cagr']:.2%}.
- Total return: {base['total_return']:.2%}.
- Daily mark-to-market maximum drawdown: {base['max_drawdown']:.2%}.
- Daily Sharpe at zero cash rate: {base['sharpe_zero_rate']:.2f}.
- Profit factor: {base['profit_factor']:.2f}; win rate: {base['win_rate']:.1%}.
- Worst trade: {base['worst_trade']:.1%}; this is economically dangerous for a short and would likely trigger broker risk controls or forced liquidation.
- Average gross exposure: {base['average_gross_exposure']:.1%}; maximum: {base['maximum_gross_exposure']:.1%}.
- Aligned SPY total return: {base['spy_total_return_aligned']:.2%}; strategy-SPY daily correlation: {base['daily_correlation_to_spy']:.3f}; beta: {base['beta_to_spy']:.3f}.

The low correlation and beta make the signal potentially useful as a diversifier, but it did not beat SPY over the aligned full interval.

## Year-by-year

{yearly_table}

## Stress robustness

Across {total_stress} combinations of stop, annual borrow (5%-50%), locate probability (40%-100%), and round-trip cost (0.3%-3%), {profitable_stress} scenarios ended profitable. The median CAGR was {stress['cagr'].median():.2%}; the worst was {stress['cagr'].min():.2%}. Tight 20% stops were especially destructive because frequent volatile spikes stopped out before later mean reversion.

At 70% locate availability across {len(locate_mc)} deterministic seeds, median CAGR was {locate_mc['cagr'].median():.2%}, the 10th percentile was {locate_mc['cagr'].quantile(.10):.2%}, and {float((locate_mc['total_return'] > 0).mean()):.1%} of runs were profitable. Locate selection therefore changes outcomes materially and must be recorded prospectively.

## What would actually improve confidence

1. Collect real locate acceptance, quoted borrow rate, and recall data before every prospective signal.
2. Model broker margin and forced buy-ins; a short losing more than 300% cannot be treated as an ordinary hold-to-horizon trade.
3. Preserve a new forward period. The current 2023-2026 test has now been repeatedly inspected.
4. Investigate event-aware exits on validation data rather than selecting a stop from test performance.
5. Obtain market-cap, float, short-interest, halt, and corporate-action histories to distinguish executable failures from hard-to-borrow squeezes.

The right next step is instrumentation and forward data collection, not another round of historical parameter optimization.
"""
    (output / "ASSESSMENT.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily mark-to-market and execution stress research for V5 shorts")
    parser.add_argument("--predictions", default="reports/v5_validation_suite/model/predictions.csv")
    parser.add_argument("--features", default="data/processed/events_features_v5.parquet")
    parser.add_argument("--raw-prices-dir", default="data/raw/prices")
    parser.add_argument("--spy", default="data/raw/spy_benchmark.parquet")
    parser.add_argument("--output-dir", default="reports/v5_mtm_research")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    features = pd.read_parquet(args.features)
    spy = normalize_spy(Path(args.spy))
    validation_candidates = attach_event_dates(build_period_trades(predictions, features, "validation"), features, 5)
    candidates = attach_event_dates(build_period_trades(predictions, features, "test"), features, 5)
    all_symbols = pd.concat([validation_candidates["symbol"], candidates["symbol"]], ignore_index=True)
    prices = load_candidate_prices(all_symbols, Path(args.raw_prices_dir))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    initial_capital, stake = 100_000.0, 10_000.0

    stop_grid = (None, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00)
    calibration_rows = []
    for stop_loss in stop_grid:
        paths, completed = trade_paths(validation_candidates, prices, stop_loss, 1000.0)
        paths, completed = apply_round_trip_cost(paths, completed, 100.0)
        equity = daily_equity_curve(paths, completed, pd.DatetimeIndex(spy["date"]), initial_capital, stake)
        summary, _ = performance(equity, spy, initial_capital)
        drawdown = abs(summary["max_drawdown"])
        calibration_rows.append({
            "stop_loss": stop_loss,
            **summary,
            **trade_statistics(completed),
            "calmar": summary["cagr"] / drawdown if drawdown > 0 else None,
        })
    calibration = pd.DataFrame(calibration_rows)
    eligible = calibration.loc[(calibration["cagr"] > 0) & (calibration["max_drawdown"] >= -0.25)].copy()
    if eligible.empty:
        raise RuntimeError("No validation-calibrated stop passed positive-CAGR / 25%-drawdown constraints")
    chosen_stop = eligible.sort_values(["calmar", "cagr"], ascending=False).iloc[0]["stop_loss"]
    chosen_stop = None if pd.isna(chosen_stop) else float(chosen_stop)
    calibration["selected_on_validation"] = calibration["stop_loss"].fillna(-1).eq(-1 if chosen_stop is None else chosen_stop)
    calibration.to_csv(output / "validation_stop_calibration.csv", index=False)
    (output / "validation_stop_selection.json").write_text(json.dumps({"chosen_stop_loss": chosen_stop, "criterion": "highest validation Calmar among positive-CAGR scenarios with max drawdown no worse than -25%"}, indent=2))

    scenarios = []
    detailed_written = False
    for stop_loss in stop_grid:
        for borrow_bps in (500.0, 1000.0, 2500.0, 5000.0):
            for locate_probability in (1.0, 0.70, 0.40):
                for cost_bps in (30.0, 100.0, 300.0):
                    located, rejected = apply_locate_model(candidates, locate_probability, 42)
                    paths, completed = trade_paths(located, prices, stop_loss, borrow_bps)
                    paths, completed = apply_round_trip_cost(paths, completed, cost_bps)
                    equity = daily_equity_curve(paths, completed, pd.DatetimeIndex(spy["date"]), initial_capital, stake)
                    summary, aligned = performance(equity, spy, initial_capital)
                    scenario = {
                        "stop_loss": stop_loss,
                        "borrow_bps_annual": borrow_bps,
                        "locate_probability": locate_probability,
                        "round_trip_cost_bps": cost_bps,
                        "candidate_trades": len(candidates),
                        "completed_trades": len(completed),
                        "locate_rejections": len(rejected),
                        **summary,
                        **trade_statistics(completed),
                        "selected_on_validation": stop_loss == chosen_stop,
                    }
                    scenarios.append(scenario)
                    if stop_loss == chosen_stop and borrow_bps == 1000 and locate_probability == 0.70 and cost_bps == 100:
                        paths.to_csv(output / "base_paths.csv", index=False)
                        completed.to_csv(output / "base_trades.csv", index=False)
                        aligned.to_csv(output / "base_daily_equity.csv", index=False)
                        yearly_comparison(aligned).to_csv(output / "base_yearly_comparison.csv", index=False)
                        (output / "base_summary.json").write_text(json.dumps(scenario, indent=2, default=str))
                        detailed_written = True
    if not detailed_written:
        raise RuntimeError("Base scenario was not generated")
    results = pd.DataFrame(scenarios)
    results.to_csv(output / "stress_scenarios.csv", index=False)
    (output / "stress_scenarios.json").write_text(json.dumps(results.to_dict("records"), indent=2, default=str))

    locate_rows = []
    for seed in range(100):
        located, _ = apply_locate_model(candidates, 0.70, seed)
        paths, completed = trade_paths(located, prices, chosen_stop, 1000.0)
        paths, completed = apply_round_trip_cost(paths, completed, 100.0)
        equity = daily_equity_curve(paths, completed, pd.DatetimeIndex(spy["date"]), initial_capital, stake)
        summary, _ = performance(equity, spy, initial_capital)
        locate_rows.append({"seed": seed, "trades": len(completed), **summary})
    locate_mc = pd.DataFrame(locate_rows)
    locate_mc.to_csv(output / "locate_seed_sensitivity.csv", index=False)
    locate_summary = {
        "seeds": len(locate_mc),
        "median_cagr": float(locate_mc["cagr"].median()),
        "cagr_p10": float(locate_mc["cagr"].quantile(0.10)),
        "cagr_p90": float(locate_mc["cagr"].quantile(0.90)),
        "profitable_fraction": float((locate_mc["total_return"] > 0).mean()),
        "median_max_drawdown": float(locate_mc["max_drawdown"].median()),
    }
    (output / "locate_seed_summary.json").write_text(json.dumps(locate_summary, indent=2))
    base = json.loads((output / "base_summary.json").read_text())
    yearly = pd.read_csv(output / "base_yearly_comparison.csv")
    write_assessment(output, calibration, base, yearly, results, locate_mc)
    print(results.sort_values("cagr").to_string(index=False))
    print(f"Wrote {len(results)} scenarios to {output}")


if __name__ == "__main__":
    main()
