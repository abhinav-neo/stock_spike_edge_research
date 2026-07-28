from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.train_predictive_model import build_pipeline, select_features


def evaluate(actual: pd.Series, predicted: np.ndarray) -> dict:
    a = actual.to_numpy(float)
    p = np.asarray(predicted, dtype=float)
    valid = np.isfinite(a) & np.isfinite(p)
    a = a[valid]
    p = p[valid]
    correlation = (
        float(np.corrcoef(a, p)[0, 1])
        if len(a) > 1 and np.std(a) > 0 and np.std(p) > 0
        else np.nan
    )
    return {
        "n": int(len(a)),
        "actual_mean": float(np.mean(a)) if len(a) else np.nan,
        "predicted_mean": float(np.mean(p)) if len(p) else np.nan,
        "mae": float(mean_absolute_error(a, p)) if len(a) else np.nan,
        "rmse": float(np.sqrt(mean_squared_error(a, p))) if len(a) else np.nan,
        "correlation": correlation,
        "directional_accuracy": float(np.mean(np.sign(a) == np.sign(p))) if len(a) else np.nan,
    }


def rank_metrics(actual: pd.Series, predicted: np.ndarray, fraction: float) -> dict:
    frame = pd.DataFrame({"actual": actual.to_numpy(float), "predicted": predicted}).dropna()
    ordered = frame.sort_values("predicted", ascending=False)
    count = max(1, int(np.floor(len(ordered) * fraction)))
    top = ordered.head(count)
    bottom = ordered.tail(count)
    return {
        "selection_fraction": fraction,
        "top_n": int(len(top)),
        "top_avg_return": float(top["actual"].mean()),
        "top_median_return": float(top["actual"].median()),
        "top_win_rate": float((top["actual"] > 0).mean()),
        "bottom_n": int(len(bottom)),
        "bottom_avg_return": float(bottom["actual"].mean()),
        "bottom_short_avg_return": float(-bottom["actual"].mean()),
        "bottom_short_win_rate": float((bottom["actual"] < 0).mean()),
        "top_bottom_spread": float(top["actual"].mean() - bottom["actual"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanding-window walk-forward validation for V5")
    parser.add_argument("--input", default="data/processed/events_features_v5.parquet")
    parser.add_argument("--target", default="forward_return_5d")
    parser.add_argument("--model", choices=["hist_gradient_boosting", "random_forest"], default="random_forest")
    parser.add_argument("--first-test-year", type=int, default=2020)
    parser.add_argument("--last-test-year", type=int, default=None)
    parser.add_argument("--minimum-train-events", type=int, default=150)
    parser.add_argument("--selection-fraction", type=float, default=0.10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--importance-repeats", type=int, default=5)
    parser.add_argument("--output-dir", default="reports/v5_walk_forward")
    args = parser.parse_args()

    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError(f"Feature file not found: {source}")
    data = pd.read_parquet(source)
    if args.target not in data.columns or "event_date" not in data.columns:
        raise ValueError("Input must contain event_date and the selected target")

    data = data.loc[data[args.target].notna()].copy()
    data["event_date"] = pd.to_datetime(data["event_date"])
    data["year"] = data["event_date"].dt.year
    features = select_features(data.drop(columns=["year"]), args.target)
    available_years = sorted(data["year"].unique())
    last_year = args.last_test_year or max(available_years)
    test_years = [year for year in available_years if args.first_test_year <= year <= last_year]

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []

    for year in test_years:
        train = data.loc[data["year"] < year].copy()
        test = data.loc[data["year"] == year].copy()
        if len(train) < args.minimum_train_events or test.empty:
            fold_rows.append({
                "test_year": int(year),
                "train_events": int(len(train)),
                "test_events": int(len(test)),
                "status": "skipped_insufficient_data",
            })
            continue

        model = build_pipeline(train[features], args.model, args.random_state + int(year))
        model.fit(train[features], train[args.target])
        predicted = model.predict(test[features])
        metrics = evaluate(test[args.target], predicted)
        metrics.update(rank_metrics(test[args.target], predicted, args.selection_fraction))
        metrics.update({
            "test_year": int(year),
            "train_start": str(train["event_date"].min().date()),
            "train_end": str(train["event_date"].max().date()),
            "train_events": int(len(train)),
            "test_events": int(len(test)),
            "status": "completed",
        })
        fold_rows.append(metrics)

        predictions = test[["symbol", "event_date", args.target]].copy()
        predictions["predicted_return"] = predicted
        predictions["test_year"] = year
        predictions["prediction_rank_pct"] = predictions["predicted_return"].rank(pct=True)
        prediction_frames.append(predictions)

        if args.importance_repeats > 0 and len(test) >= 20:
            importance = permutation_importance(
                model,
                test[features],
                test[args.target],
                n_repeats=args.importance_repeats,
                random_state=args.random_state + int(year),
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
            )
            importance_frames.append(pd.DataFrame({
                "test_year": year,
                "feature": features,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }))

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(output / "walk_forward_metrics.csv", index=False)
    predictions_all = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    predictions_all.to_csv(output / "walk_forward_predictions.csv", index=False)

    if importance_frames:
        importance_all = pd.concat(importance_frames, ignore_index=True)
        importance_all.to_csv(output / "permutation_importance_by_year.csv", index=False)
        importance_summary = (
            importance_all.groupby("feature", as_index=False)
            .agg(
                years=("test_year", "nunique"),
                mean_importance=("importance_mean", "mean"),
                median_importance=("importance_mean", "median"),
                positive_year_fraction=("importance_mean", lambda x: float((x > 0).mean())),
            )
            .sort_values(["mean_importance", "positive_year_fraction"], ascending=False)
        )
        importance_summary.to_csv(output / "permutation_importance_summary.csv", index=False)
    else:
        importance_summary = pd.DataFrame()

    completed = folds.loc[folds.get("status", "") == "completed"].copy() if not folds.empty else pd.DataFrame()
    aggregate = {
        "model": args.model,
        "target": args.target,
        "features": features,
        "feature_count": len(features),
        "folds_requested": len(test_years),
        "folds_completed": int(len(completed)),
        "mean_correlation": float(completed["correlation"].mean()) if len(completed) else np.nan,
        "median_correlation": float(completed["correlation"].median()) if len(completed) else np.nan,
        "positive_correlation_fraction": float((completed["correlation"] > 0).mean()) if len(completed) else np.nan,
        "mean_top_return": float(completed["top_avg_return"].mean()) if len(completed) else np.nan,
        "mean_bottom_return": float(completed["bottom_avg_return"].mean()) if len(completed) else np.nan,
        "mean_top_bottom_spread": float(completed["top_bottom_spread"].mean()) if len(completed) else np.nan,
        "warning": "Walk-forward folds are out-of-sample by year, but repeated research decisions based on these results can still create selection bias.",
    }
    (output / "walk_forward_summary.json").write_text(json.dumps(aggregate, indent=2, default=str))

    print("\nWalk-forward metrics")
    print(folds.to_string(index=False))
    if not importance_summary.empty:
        print("\nTop permutation features")
        print(importance_summary.head(20).to_string(index=False))
    print("\nAggregate")
    print(json.dumps(aggregate, indent=2, default=str))
    print(f"\nWrote walk-forward outputs to {output}")


if __name__ == "__main__":
    main()
