# Alternative Research Final Decision

## Decision

**No strategy in the current research space is acceptable for paper trading. Paper trading remains disabled.**

This is the proper output of the completed research: the ranking signal exists, but no tested implementation converts it into a sufficiently safe and benchmark-competitive portfolio without violating the project's risk constraints.

## Evidence

### V5 failed-spike short

- Realistic daily-MTM base: 15.68% CAGR, -20.55% drawdown, 1.03 Sharpe, and 2.13 profit factor.
- Aligned SPY returned 71.52% versus the strategy's 59.69%.
- Worst individual short return: -322.1%.
- Conventional 20%-100% stops all lost money on 2020-2022 validation data.
- The signal is diversifying, but its uncontrolled squeeze risk is unacceptable.

### Generalized Alpha Factory

- 672 candidates evaluated across gap fade, momentum, mean reversion, and breakout families.
- 57 train discoveries passed FDR, 22 survived validation, and 18 survived locked test.
- All 18 locked survivors were short strategies: 10 gap-fade and 8 mean-reversion variants.
- The constrained four-representative portfolio produced 8.14% CAGR, 24.79% drawdown, 0.54 Sharpe, and a -58.89% worst trade.
- No long strategy survived the locked validation funnel.

### Gross-capped SPY overlay

- A superficially attractive overlay requires more than 100% gross exposure.
- When S&P exposure is reduced to reserve short capacity, every nonzero overlay size lowers validation CAGR and Calmar.
- Validation selects no overlay.

### Gross-capped long/short V5 overlay

- Position sizes from 0.5% through 2.0% were tested with a 70% short-locate scenario, 10% annual borrow, 1% round-trip costs, and explicit S&P-capital reservation.
- Every nonzero size reduced validation CAGR and Sharpe versus SPY.
- Validation again selects a 0% overlay.

## Non-negotiable conclusion

Continuing to tune thresholds, stops, position sizes, or model settings against the inspected 2015-2026 history would manufacture backtest performance. It would not produce new evidence. The current historical dataset has been exhausted for honest model selection.

An acceptable next candidate requires genuinely new information or a genuinely new forward sample:

1. Historical or prospectively captured locate decisions, quoted borrow fees, recalls, and forced buy-ins.
2. Market capitalization, float, short interest, halt, and corporate-action histories.
3. A new, untouched forward period that is not used for parameter selection.
4. A bounded-loss instrument with reliable historical quotes, such as sufficiently liquid option spreads, if options are part of the intended mandate.
5. New independent signal families supported by point-in-time data rather than further transformations of the same OHLCV panel.

Until at least one of those inputs exists, the benchmark allocation is the only accepted portfolio in this research: no alpha overlay and no paper trades.

## V6 free-data follow-up

SPY regime, VIX regime, inferred sector-relative features, and an alternate gradient-boosting model were subsequently tested. None improved both validation and walk-forward behavior; gradient boosting materially overfit. A leakage-safe point-in-time interface is ready for future historical market-cap, float, short-interest, borrow, halt, or fundamental data. The final decision is unchanged. See `reports/v6_improvement/ASSESSMENT.md`.

## Free FINRA short-volume follow-up

Free FINRA daily short-sale volume was integrated with 83.5% exact symbol/event-date coverage. A ratio-only feature modestly improved validation correlation (0.0700 versus 0.0661) and mean yearly walk-forward correlation (0.1611 versus 0.1534), while worsening test correlation and yearly spread. The strict portfolio improved from 21.11% to 22.27% CAGR but contained a -321.70% short loss on allocated notional. FINRA transaction volume is not borrow availability or short interest. The feature is retained for research, but the zero-allocation and no-paper-trading decision is unchanged.

## Free SEC fails-to-deliver follow-up

All 274 required SEC half-month archives were integrated using conservative publication availability rather than settlement-date hindsight. SEC FTD alone worsened validation correlation and yearly spread. Combining FTD with the FINRA ratio improved test and average walk-forward correlation but did not improve validation and remained below V5 on yearly spread. FTD thresholds also failed to identify the catastrophic SMX short without test-informed tuning. The variants are rejected; accepted allocation remains zero.

## Complete free-data factorial follow-up

The full power set of SPY, VIX, inferred sector, FINRA ratio, and SEC FTD groups was evaluated with random forest and histogram gradient boosting: 32 combinations and 64 validation-only variants. The best random forest improved validation correlation by only 0.0098, below the locked 0.0200 promotion gate. The best gradient-boosting result had a 0.8929 train-validation gap and was rejected as severe overfit. No variant qualified for test or portfolio promotion. Further recombination of these inputs is closed.

## Causal Markov-regime follow-up

A four-state, online Markov chain based on strictly historical SPY momentum and volatility was evaluated without consulting locked-test outcomes. Markov features reduced random-forest validation correlation from 0.0661 to 0.0636. Histogram gradient boosting improved by only 0.0053 while retaining a 0.9090 train-validation gap. Neither model passed the locked +0.0200 improvement gate. No portfolio run was authorized and the accepted allocation remains zero. See `reports/markov_regime/ASSESSMENT.md`.
