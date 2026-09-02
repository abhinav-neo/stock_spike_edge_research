from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.markov_regime_features import _past_percentile


def normalize_bars(raw: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw.columns, pd.MultiIndex):
        required = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}
        missing = required - set(raw.columns)
        if missing:
            raise ValueError(f"bars missing required columns: {sorted(missing)}")
        result = raw.copy()
    else:
        pieces = []
        tickers = raw.columns.get_level_values(1).unique()
        for ticker in tickers:
            frame = raw.xs(ticker, axis=1, level=1).copy().reset_index()
            timestamp = frame.columns[0]
            frame = frame.rename(columns={timestamp: "timestamp"})
            frame.columns = [str(column).lower() for column in frame.columns]
            frame["symbol"] = str(ticker)
            pieces.append(frame)
        result = pd.concat(pieces, ignore_index=True)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    numeric = ["open", "high", "low", "close", "volume"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    result = result.dropna(subset=["timestamp", "symbol", "open", "close"])
    if result.duplicated(["symbol", "timestamp"]).any():
        raise ValueError("bars contain duplicate symbol/timestamp rows")
    return result.sort_values(["timestamp", "symbol"]).reset_index(drop=True)


def add_causal_regime(bars: pd.DataFrame, benchmark: str = "SPY", minimum_history: int = 100) -> pd.DataFrame:
    benchmark_bars = bars.loc[bars["symbol"] == benchmark].sort_values("timestamp").copy()
    close = benchmark_bars["close"]
    returns = close.pct_change()
    volatility = returns.rolling(12, min_periods=12).std()
    vol_percentile = _past_percentile(volatility, minimum_history)
    momentum = close.pct_change(12)
    state = pd.Series("unavailable", index=benchmark_bars.index, dtype=object)
    ready = vol_percentile.notna() & momentum.notna()
    state.loc[ready] = "quiet"
    state.loc[ready & (vol_percentile >= 0.80)] = "stress"
    state.loc[ready & (vol_percentile < 0.80) & (momentum.abs() > volatility * np.sqrt(12))] = "trend"
    regimes = pd.DataFrame({"timestamp": benchmark_bars["timestamp"], "regime": state})
    return bars.merge(regimes, on="timestamp", how="left", validate="many_to_one")


def candidate_grid() -> list[dict]:
    return [
        {"mode": mode, "book": book, "lookback": lookback, "holding": holding, "regime": regime, "top_k": top_k}
        for mode, book, lookback, holding, regime, top_k in itertools.product(
            ("reversal", "momentum"),
            ("long_only", "long_short"),
            (1, 3, 6),
            (1, 3, 6),
            ("all", "quiet", "trend", "stress"),
            (1, 2),
        )
    ]


def strategy_returns(
    bars: pd.DataFrame,
    candidate: dict,
    round_trip_cost_bps: float = 10.0,
    benchmark: str = "SPY",
    matrices: tuple[pd.DataFrame, pd.DataFrame, pd.Series] | None = None,
) -> pd.DataFrame:
    if matrices is None:
        opens = bars.pivot(index="timestamp", columns="symbol", values="open").sort_index()
        closes = bars.pivot(index="timestamp", columns="symbol", values="close").sort_index()
        regime = bars.drop_duplicates("timestamp").set_index("timestamp")["regime"].reindex(opens.index)
    else:
        opens, closes, regime = matrices
    assets = [column for column in opens.columns if column != benchmark]
    scores = closes[assets].pct_change(int(candidate["lookback"]), fill_method=None)
    if benchmark in closes:
        scores = scores.sub(closes[benchmark].pct_change(int(candidate["lookback"]), fill_method=None), axis=0)
    holding = int(candidate["holding"])
    future = opens[assets].shift(-(holding + 1)).div(opens[assets].shift(-1)).sub(1.0)
    score_values = scores.to_numpy(dtype=float)
    future_values = future.to_numpy(dtype=float)
    regime_values = regime.to_numpy(dtype=object)
    rows = []
    for position in range(0, len(opens) - holding - 1, holding):
        timestamp = opens.index[position]
        current_regime = regime_values[position]
        if candidate["regime"] != "all" and current_regime != candidate["regime"]:
            continue
        score = score_values[position]
        realized = future_values[position]
        valid = np.isfinite(score) & np.isfinite(realized)
        if np.count_nonzero(valid) < 2 * int(candidate["top_k"]):
            continue
        score = score[valid]
        realized = realized[valid]
        ascending = candidate["mode"] == "reversal"
        order = np.argsort(score)
        if not ascending:
            order = order[::-1]
        long_positions = order[: int(candidate["top_k"])]
        gross_return = float(realized[long_positions].mean())
        legs = len(long_positions)
        if candidate["book"] == "long_short":
            short_positions = order[-int(candidate["top_k"]):]
            gross_return = 0.5 * gross_return - 0.5 * float(realized[short_positions].mean())
            legs += len(short_positions)
        cost = round_trip_cost_bps / 10_000.0
        net_return = gross_return - cost
        rows.append(
            {
                "timestamp": timestamp,
                "exit_timestamp": opens.index[position + holding + 1],
                "regime": current_regime,
                "gross_return": gross_return,
                "net_return": net_return,
                "trade_legs": legs,
            }
        )
    return pd.DataFrame(rows)


def performance(returns: pd.DataFrame) -> dict:
    if returns.empty:
        return {"cagr": np.nan, "max_drawdown": np.nan, "trades_per_day": 0.0, "observations": 0}
    equity = (1.0 + returns["net_return"]).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    elapsed_days = max((returns["exit_timestamp"].iloc[-1] - returns["timestamp"].iloc[0]).total_seconds() / 86_400, 1.0)
    cagr = float(equity.iloc[-1] ** (365.25 / elapsed_days) - 1.0) if equity.iloc[-1] > 0 else np.nan
    daily_count = max(returns["timestamp"].dt.normalize().nunique(), 1)
    std = float(returns["net_return"].std(ddof=1))
    return {
        "cagr": cagr,
        "total_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "sharpe_per_observation": float(returns["net_return"].mean() / std) if std > 0 else np.nan,
        "trades_per_day": float(returns["trade_legs"].sum() / daily_count),
        "observations": int(len(returns)),
        "trade_legs": int(returns["trade_legs"].sum()),
    }


def evaluate_period(returns: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    selected = returns.loc[(returns["timestamp"] >= start) & (returns["timestamp"] < end)].copy()
    return performance(selected)


def run_study(bars: pd.DataFrame, cost_bps: float = 10.0) -> tuple[pd.DataFrame, dict]:
    dates = pd.Index(sorted(bars["timestamp"].dt.normalize().unique()))
    if len(dates) < 45:
        raise ValueError("at least 45 trading days are required for provisional train/validation/test splits")
    train_end, validation_end = pd.Timestamp(dates[int(len(dates) * 0.50)]), pd.Timestamp(dates[int(len(dates) * 0.75)])
    end = pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)
    rows = []
    cached: dict[int, pd.DataFrame] = {}
    opens = bars.pivot(index="timestamp", columns="symbol", values="open").sort_index()
    closes = bars.pivot(index="timestamp", columns="symbol", values="close").sort_index()
    regime = bars.drop_duplicates("timestamp").set_index("timestamp")["regime"].reindex(opens.index)
    matrices = (opens, closes, regime)
    for number, candidate in enumerate(candidate_grid()):
        returns = strategy_returns(bars, candidate, cost_bps, matrices=matrices)
        cached[number] = returns
        train = evaluate_period(returns, pd.Timestamp(dates[0]), train_end)
        rows.append({"candidate_id": number, **candidate, **{f"train_{key}": value for key, value in train.items()}})
    results = pd.DataFrame(rows)
    shortlist = results.loc[results["train_observations"] >= 30].sort_values(
        ["train_cagr", "train_max_drawdown"], ascending=[False, False]
    ).head(12)
    validation_rows = []
    for _, row in shortlist.iterrows():
        metrics = evaluate_period(cached[int(row["candidate_id"])], train_end, validation_end)
        validation_rows.append({"candidate_id": int(row["candidate_id"]), **{f"validation_{key}": value for key, value in metrics.items()}})
    validation = pd.DataFrame(validation_rows)
    results = results.merge(validation, on="candidate_id", how="left")
    eligible = results.loc[
        (results["validation_cagr"] >= 0.50)
        & (results["validation_max_drawdown"] >= -0.25)
        & (results["validation_trades_per_day"] >= 5.0)
        & (results["validation_observations"] >= 20)
    ].sort_values(["validation_cagr", "validation_max_drawdown"], ascending=[False, False])
    locked = eligible.iloc[0] if not eligible.empty else None
    test_metrics = None
    if locked is not None:
        test_metrics = evaluate_period(cached[int(locked["candidate_id"])], validation_end, end)
    summary = {
        "status": "provisional_bar_data_only_not_production_eligible",
        "bars": int(len(bars)),
        "symbols": int(bars["symbol"].nunique()),
        "trading_days": int(len(dates)),
        "train_end_exclusive": str(train_end),
        "validation_end_exclusive": str(validation_end),
        "round_trip_cost_bps": cost_bps,
        "candidate_count": len(candidate_grid()),
        "validation_shortlist_count": int(len(shortlist)),
        "eligible_count": int(len(eligible)),
        "locked_candidate_id": int(locked["candidate_id"]) if locked is not None else None,
        "locked_test_metrics": test_metrics,
    }
    return results, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Provisional leakage-safe five-minute regime strategy study")
    parser.add_argument("--bars", default="data/raw/intraday_5m_provisional.parquet")
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument("--output-dir", default="reports/intraday_provisional")
    args = parser.parse_args()
    bars = add_causal_regime(normalize_bars(pd.read_parquet(args.bars)))
    results, summary = run_study(bars, args.cost_bps)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "candidate_metrics.csv", index=False)
    (output / "selection_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(results.sort_values("validation_cagr", ascending=False).head(20).to_string(index=False))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
