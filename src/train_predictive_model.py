from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NON_FEATURE_COLUMNS = {
    "symbol",
    "event_date",
    "entry_date",
    "entry_open",
    "status",
    "side",
    "sample_size_pass",
    "stable_positive",
    "target_proxy_pass",
    "eligible_for_ranking",
    "rejection_reason",
    "ranking_uses_test_period",
}


def chronological_split(
    data: pd.DataFrame,
    train_end: str,
    validation_end: str,
    target_horizon_days: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = pd.to_datetime(data["event_date"])
    train_cutoff = pd.Timestamp(train_end)
    validation_cutoff = pd.Timestamp(validation_end)

    label_start = (
        pd.to_datetime(data["entry_date"])
        if "entry_date" in data.columns
        else dates + pd.offsets.BDay(1)
    )
    label_end = label_start + pd.offsets.BDay(target_horizon_days)

    # Purge observations whose forward outcome crosses a split boundary.  A
    # chronological row split alone leaks future-period prices through labels.
    train = data.loc[(dates <= train_cutoff) & (label_end <= train_cutoff)].copy()
    validation = data.loc[
        (dates > train_cutoff)
        & (dates <= validation_cutoff)
        & (label_end <= validation_cutoff)
    ].copy()
    test = data.loc[dates > validation_cutoff].copy()
    return train, validation, test


def target_horizon(target: str) -> int:
    match = re.fullmatch(r"forward_return_(\d+)d", target)
    if not match:
        raise ValueError(f"Cannot infer forward horizon from target: {target}")
    return int(match.group(1))


def select_features(data: pd.DataFrame, target: str) -> list[str]:
    forbidden = set(NON_FEATURE_COLUMNS)
    forbidden.add(target)
    forbidden.update(column for column in data.columns if column.startswith("forward_return_"))
    forbidden.update(column for column in data.columns if column.startswith("gain_retention_"))
    forbidden.update(column for column in data.columns if column.startswith("above_entry_open_"))
    forbidden.update(column for column in data.columns if column.startswith("max_forward_"))
    forbidden.update(column for column in data.columns if column.startswith("days_to_"))
    forbidden.update(column for column in data.columns if column.startswith("consecutive_days_"))

    return [column for column in data.columns if column not in forbidden]


def build_pipeline(frame: pd.DataFrame, model_name: str, random_state: int) -> Pipeline:
    numeric_columns = frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_columns = [column for column in frame.columns if column not in numeric_columns]

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if model_name == "random_forest":
        model = RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=10,
            max_features="sqrt",
            n_jobs=-1,
            random_state=random_state,
        )
    else:
        model = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=random_state,
        )

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def evaluate_period(name: str, actual: pd.Series, predicted: np.ndarray) -> dict:
    actual_values = actual.to_numpy(float)
    predicted_values = np.asarray(predicted, dtype=float)
    valid = np.isfinite(actual_values) & np.isfinite(predicted_values)
    actual_values = actual_values[valid]
    predicted_values = predicted_values[valid]

    if len(actual_values) == 0:
        return {"period": name, "n": 0}

    correlation = (
        float(np.corrcoef(actual_values, predicted_values)[0, 1])
        if len(actual_values) > 1
        and np.std(actual_values) > 0
        and np.std(predicted_values) > 0
        else np.nan
    )
    directional_accuracy = float(np.mean(np.sign(actual_values) == np.sign(predicted_values)))

    return {
        "period": name,
        "n": int(len(actual_values)),
        "actual_mean": float(np.mean(actual_values)),
        "predicted_mean": float(np.mean(predicted_values)),
        "mae": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "correlation": correlation,
        "directional_accuracy": directional_accuracy,
    }


def prediction_frame(
    source: pd.DataFrame,
    predicted: np.ndarray,
    target: str,
    period: str,
) -> pd.DataFrame:
    result = source[["symbol", "event_date", target]].copy()
    result["predicted_return"] = predicted
    result["period"] = period
    result["prediction_rank_pct"] = result["predicted_return"].rank(pct=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a V5 chronological baseline model")
    parser.add_argument("--input", default="data/processed/events_features_v5.parquet")
    parser.add_argument("--target", default="forward_return_5d")
    parser.add_argument("--train-end", default="2019-12-31")
    parser.add_argument("--validation-end", default="2022-12-31")
    parser.add_argument(
        "--model",
        choices=["hist_gradient_boosting", "random_forest"],
        default="hist_gradient_boosting",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output-dir", default="reports/v5_model")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Feature file not found: {input_path}")

    data = pd.read_parquet(input_path)
    if args.target not in data.columns:
        raise ValueError(f"Target column not found: {args.target}")
    if "event_date" not in data.columns:
        raise ValueError("Input must contain event_date")

    data = data.loc[data[args.target].notna()].copy()
    data["event_date"] = pd.to_datetime(data["event_date"])
    horizon = target_horizon(args.target)
    train, validation, test = chronological_split(
        data, args.train_end, args.validation_end, target_horizon_days=horizon
    )

    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Chronological split produced an empty period. Adjust split dates or obtain more history."
        )

    features = select_features(data, args.target)
    if not features:
        raise ValueError("No usable feature columns remain after leakage exclusions")

    pipeline = build_pipeline(train[features], args.model, args.random_state)
    pipeline.fit(train[features], train[args.target])

    predictions = {
        "train": pipeline.predict(train[features]),
        "validation": pipeline.predict(validation[features]),
        "test": pipeline.predict(test[features]),
    }

    metrics = [
        evaluate_period("train", train[args.target], predictions["train"]),
        evaluate_period("validation", validation[args.target], predictions["validation"]),
        evaluate_period("test", test[args.target], predictions["test"]),
    ]

    combined_predictions = pd.concat(
        [
            prediction_frame(train, predictions["train"], args.target, "train"),
            prediction_frame(validation, predictions["validation"], args.target, "validation"),
            prediction_frame(test, predictions["test"], args.target, "test"),
        ],
        ignore_index=True,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(output_dir / "model_metrics.csv", index=False)
    combined_predictions.to_csv(output_dir / "predictions.csv", index=False)

    summary = {
        "model": args.model,
        "target": args.target,
        "train_end": args.train_end,
        "validation_end": args.validation_end,
        "feature_count": len(features),
        "features": features,
        "metrics": metrics,
        "warning": (
            "This is a predictive research baseline, not a trading strategy. "
            "Do not use test results for model selection; validate ranked signals in a realistic portfolio simulator."
        ),
    }
    (output_dir / "model_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"Wrote model outputs to {output_dir}")


if __name__ == "__main__":
    main()
