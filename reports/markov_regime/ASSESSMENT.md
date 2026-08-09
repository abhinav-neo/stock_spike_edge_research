# Causal Markov-Regime Research Assessment

## Verdict

The first Markov-regime challenger is rejected. It does not qualify for locked-test,
portfolio, or paper-trading promotion. Accepted alpha allocation remains zero.

## Method

SPY observations were classified causally into quiet-bear, neutral, quiet-bull,
and stress states using 20-day momentum and volatility percentiles computed against
strictly prior observations. A first-order transition matrix was updated online with
Laplace smoothing. Each event therefore received only state and transition information
available at its event close, before the next-open entry assumed by the strategy.

Future-invariance tests alter all prices after a cutoff and verify that earlier states
do not change. A second test changes every locked-test target and verifies that model
selection remains unchanged.

## Validation-only results

| Variant | Model | Train correlation | Validation correlation | Gap | Improvement |
|---|---:|---:|---:|---:|---:|
| V5 baseline | Random forest | 0.6738 | 0.0661 | 0.6077 | -- |
| Markov | Random forest | 0.6861 | 0.0636 | 0.6225 | -0.0026 |
| V5 baseline | Histogram gradient boosting | 0.9799 | 0.0651 | 0.9147 | -- |
| Markov | Histogram gradient boosting | 0.9804 | 0.0714 | 0.9090 | +0.0053 |

The locked minimum improvement was +0.0200, with no more than +0.0500 excess
train-validation gap. Eligible candidates: **0**.

The gradient-boosting result remains severely overfit and its small improvement is far
below the multiplicity-aware gate. The random forest becomes worse. State-conditioned
mean returns also change sign across periods, so a hard regime filter is not stable.

## Implication for the 50% objective

No return, drawdown, or $10,000 projection is reported for this challenger because it
failed before locked-test and portfolio evaluation. Computing and optimizing those
metrics after a failed validation gate would convert the test period into training data.

The current daily-bar/event dataset can test multi-day regime conditioning, but it cannot
establish high-frequency execution performance. Honest pursuit of the full objective now
requires point-in-time intraday quotes/trades, delisting and corporate-action handling,
and a queue/slippage-aware fill model. Markov structure alone does not create an edge.
