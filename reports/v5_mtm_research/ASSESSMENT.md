# V5 Daily Mark-to-Market Research Assessment

## Verdict

The V5 failed-spike short remains a useful research signal, but the daily-path analysis weakens the earlier event-return result. It is **not ready for paper trading**. The principal problem is unbounded short-squeeze risk, not ordinary transaction cost.

## Validation-calibrated risk control

Risk controls were selected only on 2020-2022 validation trades. No tested stop from 20% through 100% was profitable on validation data; the selected rule was no stop, with validation CAGR 2.96%, drawdown -16.99%, and Calmar 0.17. This means the research does not support claiming that a conventional stop improves the strategy.

## Forward test with realistic execution assumptions

The primary 2023+ scenario uses the validation-selected risk rule, 70% deterministic locate availability, 10% annual borrow, 1% round-trip cost, fixed $10,000 notional, and at most the already capacity-controlled positions:

- Trades completed: 47 of 73 candidates.
- CAGR: 15.68%.
- Total return: 59.69%.
- Daily mark-to-market maximum drawdown: -20.55%.
- Daily Sharpe at zero cash rate: 1.03.
- Profit factor: 2.13; win rate: 70.2%.
- Worst trade: -322.1%; this is economically dangerous for a short and would likely trigger broker risk controls or forced liquidation.
- Average gross exposure: 2.9%; maximum: 30.0%.
- Aligned SPY total return: 71.52%; strategy-SPY daily correlation: -0.038; beta: -0.038.

The low correlation and beta make the signal potentially useful as a diversifier, but it did not beat SPY over the aligned full interval.

## Year-by-year

| Year | Strategy | SPY aligned | Excess |
|---:|---:|---:|---:|
| 2023 | 7.58% | 21.09% | -13.50% |
| 2024 | 25.92% | 25.59% | 0.33% |
| 2025 | -1.12% | 18.01% | -19.13% |
| 2026 | 18.26% | -3.83% | 22.08% |

## Stress robustness

Across 252 combinations of stop, annual borrow (5%-50%), locate probability (40%-100%), and round-trip cost (0.3%-3%), 193 scenarios ended profitable. The median CAGR was 7.45%; the worst was -8.52%. Tight 20% stops were especially destructive because frequent volatile spikes stopped out before later mean reversion.

At 70% locate availability across 100 deterministic seeds, median CAGR was 14.53%, the 10th percentile was 9.57%, and 100.0% of runs were profitable. Locate selection therefore changes outcomes materially and must be recorded prospectively.

## What would actually improve confidence

1. Collect real locate acceptance, quoted borrow rate, and recall data before every prospective signal.
2. Model broker margin and forced buy-ins; a short losing more than 300% cannot be treated as an ordinary hold-to-horizon trade.
3. Preserve a new forward period. The current 2023-2026 test has now been repeatedly inspected.
4. Investigate event-aware exits on validation data rather than selecting a stop from test performance.
5. Obtain market-cap, float, short-interest, halt, and corporate-action histories to distinguish executable failures from hard-to-borrow squeezes.

The right next step is instrumentation and forward data collection, not another round of historical parameter optimization.
