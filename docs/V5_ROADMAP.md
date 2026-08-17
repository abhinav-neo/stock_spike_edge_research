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

## Roadmap status - 2026-08-17

| Stage | Status | Controlling result |
|---|---|---|
| Leakage-safe features and chronological model | Complete | Signal survives as research only. |
| Ranked/yearly validation and robustness | Complete | Repeated test inspection closes further tuning on 2015-2026 data. |
| Capital-constrained daily-MTM portfolio | Complete | 15.68% CAGR, -20.55% drawdown, 1.03 Sharpe; below 40% target and SPY. |
| Broker margin and forced liquidation | Complete | Best bounded scenario: 11.64% CAGR, -13.71% drawdown; rejected. |
| Free SPY/VIX/sector/FINRA/SEC/Markov variants | Complete | None passed the locked validation-improvement gate. |
| Point-in-time SEC fundamentals interface | Implemented, source blocked | SEC endpoint returns HTTP 403 from this host; no fabricated backfill. |
| Bounded-loss option-spread path | Complete, rejected | Only 7/58 in-window events were constructible; historical quotes unavailable. |
| Corporate-action and halt evidence | Historical coverage complete; prospective capture active | 50/104 candidates halted on event day; no leakage-safe historical filter selected. |
| Broker eligibility and locate evidence | Operational | 18/18 required ETB locates confirmed; six signals rejected. |
| Forward execution/statistical validation | Collecting | 0/100 settled, 0/60 dates, 0/180 days; allocation remains 0%. |

The remaining critical path is elapsed forward evidence, not additional tuning. The
scheduled zero-capital observer must accumulate at least 100 settled observations, 60
independent dates, 180 calendar days, 95% execution and snapshot coverage, acceptable
spreads, broker eligibility/locate evidence, and the locked statistical thresholds.
Only then may the combined gate consider a paper-trading candidate. Live-money order
submission is outside this roadmap and remains disabled.

## Important limitation

The current model output is not CAGR and is not a live trading strategy. The untouched test period must not be repeatedly used to tune features or model settings.
