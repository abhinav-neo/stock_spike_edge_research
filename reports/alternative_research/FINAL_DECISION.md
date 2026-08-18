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

### Broker margin and forced liquidation

- Position-level broker liquidation was modeled independently of test returns with
  gap-aware fills under 50%/30%, 100%/50%, and 200%/100% initial/maintenance profiles.
- The Reg-T-like profile liquidated 32 of 47 located trades and produced -2.39% CAGR.
- The 100%/50% house profile produced 6.57% CAGR; the 200%/100% hard-to-borrow profile
  produced 11.64% CAGR. Neither beat aligned SPY or approached the locked 40% target.
- Forced liquidation bounded the worst modeled trade to roughly -47% to -51%, confirming
  that realistic broker intervention repairs the impossible -322% hold but destroys much
  of the apparent edge. See `reports/margin_liquidation/ASSESSMENT.md`.

### Bounded-loss put spreads

- Alpaca historical options data begins in February 2024, covering only 58 of the 73
  locked strict candidates.
- Only 7 of those 58 candidates (12.1%) had at least one 14-45 DTE expiration and two
  put strikes, the minimum topology for a vertical spread.
- The provider exposes historical option trades and bars but not historical bid/ask
  quotes; the free feed is indicative rather than actual OPRA.
- Entry debit, exit credit, spread, slippage, and fill feasibility therefore cannot be
  reconstructed honestly. The bounded-loss path is rejected without fitting returns to
  the seven optionable test events. See `reports/options_coverage/ASSESSMENT.md`.

## Non-negotiable conclusion

Continuing to tune thresholds, stops, position sizes, or model settings against the inspected 2015-2026 history would manufacture backtest performance. It would not produce new evidence. The current historical dataset has been exhausted for honest model selection.

An acceptable next candidate requires genuinely new information or a genuinely new forward sample:

1. Historical or prospectively captured locate decisions, quoted borrow fees, recalls, and forced buy-ins.
2. Market capitalization, float, short interest, halt, and corporate-action histories.
3. A new, untouched forward period that is not used for parameter selection.
4. A bounded-loss instrument with reliable historical quotes, such as sufficiently liquid option spreads, if options are part of the intended mandate.
5. New independent signal families supported by point-in-time data rather than further transformations of the same OHLCV panel.

Until at least one of those inputs exists, the benchmark allocation is the only accepted portfolio in this research: no alpha overlay and no paper trades.

## Forward observation

`src.forward_observation` records signals from the locked candidate set in an
append-only, zero-capital ledger and settles them only after the full forward horizon
is available. It never creates orders. This is the approved path for collecting a new,
untouched forward sample while allocation remains zero.

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

## Corporate-action and trading-halt follow-up

Official Alpaca corporate actions and Nasdaq historical halt records were added as an
independent risk-data coverage study. Historical corporate actions lack a guaranteed
point-in-time creation timestamp, and validation contains too few recent reverse-split
examples to select a stable exclusion window. Event-day halts are retained as execution
risk evidence, not a return-tuned filter. The unattended forward pipeline now timestamps
both sources on first capture. No historical variant is promoted and accepted allocation
remains zero. See `reports/event_risk_coverage/ASSESSMENT.md`.

The fixed rule excluding all event-day halts was then gated on validation sample size
before any test-period return calculation. It retained only 17 of 31 validation
candidates, below the locked minimum of 30. Test CAGR evaluation was therefore not
authorized. See `reports/halt_exclusion/ASSESSMENT.md`.

## FINRA consolidated short-interest follow-up

The official public FINRA API returned 9,299 semi-monthly records for the candidate
symbols. A conservative 14-calendar-day publication lag and 45-day staleness cap yielded
coverage for only 20 of 31 validation candidates, below the locked minimum of 30. No
short-interest threshold or test-period return was evaluated. The path remains available
for prospective collection but is rejected for historical promotion. See
`reports/finra_short_interest/ASSESSMENT.md`.

## Point-in-time news follow-up

Alpaca/Benzinga news was collected only for the 2015-2022 train and validation events.
Validation event-news coverage was 69.9%. A fixed article-topology group, six fixed
semantic categories, and their combined feature set were evaluated with both existing
model families. The best validation improvement was only +0.0024 versus the locked
+0.0200 gate. Test news was not collected and no test return was evaluated. See
`reports/alpaca_news_research/ASSESSMENT.md`.

## Point-in-time intraday follow-up

SIP minute bars were collected only for the 2015-2022 train and validation events, with
78.3% validation coverage of at least 30 event-day bars. Path, volume, gap, and combined
feature groups were evaluated. The best path-only variant improved validation correlation
by +0.0059, below the locked +0.0200 gate; combined intraday features worsened it. Test
minute bars were not collected and no test return was evaluated. See
`reports/alpaca_intraday_research/ASSESSMENT.md`.
