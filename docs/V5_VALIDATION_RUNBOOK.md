# Version 5 Validation Runbook

This runbook completes the pending validation work after generating a V5 prediction file.

## 1. Pull the branch

```powershell
git fetch origin
git checkout v5-feature-engineering
git pull origin v5-feature-engineering
```

## 2. Ranked prediction diagnostics

For the existing random-forest test predictions:

```powershell
python -m src.analyze_ranked_predictions `
  --input reports\v5_random_forest\predictions.csv `
  --target forward_return_5d `
  --period test `
  --output-dir reports\v5_random_forest_analysis
```

Key outputs:

- `decile_metrics.csv`
- `percentile_metrics.csv`
- `yearly_stability.csv`
- `analysis_summary.json`

The percentile report includes top and bottom 1%, 2%, 5%, 10%, and 20% groups. The yearly report checks whether ranking performance persists across calendar years.

## 3. Expanding-window walk-forward validation

Random Forest:

```powershell
python -m src.walk_forward_validation `
  --target forward_return_5d `
  --model random_forest `
  --first-test-year 2020 `
  --selection-fraction 0.10 `
  --output-dir reports\v5_walk_forward_random_forest
```

HistGradientBoosting comparison:

```powershell
python -m src.walk_forward_validation `
  --target forward_return_5d `
  --model hist_gradient_boosting `
  --first-test-year 2020 `
  --selection-fraction 0.10 `
  --output-dir reports\v5_walk_forward_hist_gradient
```

Key outputs:

- `walk_forward_metrics.csv`
- `walk_forward_predictions.csv`
- `walk_forward_summary.json`
- `permutation_importance_by_year.csv`
- `permutation_importance_summary.csv`

A candidate should not advance merely because aggregate correlation is positive. Prefer models with positive correlation in most folds, positive top-versus-bottom spread across years, and stable feature importance.

## 4. Capital-reserving portfolio approximation

Long and short test portfolios using the original test prediction file:

```powershell
python -m src.ranked_portfolio_backtest `
  --input reports\v5_random_forest\predictions.csv `
  --target forward_return_5d `
  --period test `
  --side both `
  --fraction 0.10 `
  --holding-days 5 `
  --initial-capital 100000 `
  --max-positions 10 `
  --cost-bps 30 `
  --borrow-bps-per-day 10 `
  --output-dir reports\v5_random_forest_portfolio
```

Key outputs:

- `long_trades.csv`
- `long_equity.csv`
- `short_trades.csv`
- `short_equity.csv`
- `portfolio_summary.json`

The simulator reserves capital for open positions and prevents the position count from exceeding the configured limit. Results remain approximations because the source data contains fixed-horizon event returns rather than full entry-to-exit price paths.

## 5. Recommended acceptance gate

Advance a signal to deeper robustness testing only when all of the following are true:

1. Walk-forward correlation is positive in a clear majority of completed years.
2. Top-ranked returns exceed bottom-ranked returns in most years.
3. Extreme percentile performance is not driven by one or two observations.
4. Long or short portfolio results remain positive after conservative costs.
5. Maximum drawdown is acceptable under multiple position limits and cost assumptions.
6. Important features are directionally stable across years.

Do not tune model settings against the final test period. Use walk-forward folds for research and preserve the newest available period as a final confirmation set whenever possible.
