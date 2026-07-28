# Version 5: Feature Discovery and Predictive Validation

Version 4 found no eligible candidates across 950 threshold combinations. Version 5 therefore stops expanding the same threshold grid and tests whether richer, leakage-safe information can predict post-spike returns.

## Stage 1 implemented

- `src/feature_engineering.py`
  - Builds technical, momentum, volatility, liquidity and volume features from each symbol's daily history.
  - Uses only information available through the event-day close.
  - Writes `data/processed/events_features_v5.parquet` and CSV.

- `src/train_predictive_model.py`
  - Predicts a selected forward-return horizon.
  - Excludes all forward-looking outcome columns from the feature set.
  - Uses chronological train, validation and untouched test periods.
  - Writes metrics, predictions and a machine-readable summary under `reports/v5_model/`.

## Run locally

```powershell
git checkout v5-feature-engineering
git pull origin v5-feature-engineering

python -m src.feature_engineering
python -m src.train_predictive_model --target forward_return_5d
```

Optional random-forest baseline:

```powershell
python -m src.train_predictive_model --target forward_return_5d --model random_forest --output-dir reports/v5_random_forest
```

## Outputs to inspect

- `reports/v5_model/model_metrics.csv`
- `reports/v5_model/model_summary.json`
- `reports/v5_model/predictions.csv`

The key first-pass checks are validation/test correlation and directional accuracy. A strong training result with weak validation/test results is overfitting and must be rejected.

## Acceptance gate before portfolio simulation

A model should not advance merely because one metric is positive. At minimum it should show:

1. Positive validation and untouched-test correlation.
2. Similar validation and test behavior rather than a sharp collapse.
3. Economic separation between high-ranked and low-ranked predictions after modeled costs.
4. Adequate sample size across years and market regimes.
5. No dependence on a few extreme events.

## Next stages

1. Add market context such as SPY trend and volatility regime.
2. Add sector-relative features.
3. Add fundamentals such as market capitalization, float and short interest where reliable history is available.
4. Add ranked-decile and yearly stability reports.
5. Build a chronological portfolio simulator with position overlap, capital constraints, costs and short-borrow assumptions.
6. Apply bootstrap and Monte Carlo robustness tests only to surviving signals.

## Important limitation

The current model output is not CAGR and is not a live trading strategy. The untouched test period must not be repeatedly used to tune features or model settings.
