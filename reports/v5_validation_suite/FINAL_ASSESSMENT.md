# V5 Final Assessment

## Verdict

**Not yet suitable for paper trading.** The evidence below is out-of-sample by chronological fold and uses validation-only score calibration, but this remains an event-return approximation without daily mark-to-market, borrow locates, market-cap history, halts, or executable fills.

## Walk-forward stability

Random-forest rank correlation was positive in 6 of 7 completed yearly folds; the bottom-decile short return was positive in 7 of 7 folds. That is the central stability test, not the aggregate backtest.

- 2020: correlation 0.153, bottom-decile short average 15.28%, 29 selected events.
- 2021: correlation -0.069, bottom-decile short average 7.29%, 26 selected events.
- 2022: correlation 0.211, bottom-decile short average 38.58%, 14 selected events.
- 2023: correlation 0.194, bottom-decile short average 22.35%, 18 selected events.
- 2024: correlation 0.178, bottom-decile short average 21.78%, 36 selected events.
- 2025: correlation 0.132, bottom-decile short average 14.34%, 54 selected events.
- 2026: correlation 0.275, bottom-decile short average 31.08%, 17 selected events.

## Portfolio evidence after costs

- Unfiltered short: 163 trades; CAGR 46.23%; total return 251.46%; max realized-equity drawdown -13.64%; trade-level Sharpe proxy 2.46; profit factor 2.65; win rate 73.6%.
- Minimum $5 entry price and $1M prior 20-day average dollar volume: 126 trades; CAGR 43.82%; total return 232.64%; max realized-equity drawdown -10.08%; trade-level Sharpe proxy 2.85; profit factor 3.18; win rate 77.0%.
- Minimum $10 entry price and $5M prior 20-day average dollar volume: 73 trades; CAGR 21.11%; total return 87.73%; max realized-equity drawdown -18.46%; trade-level Sharpe proxy 1.59; profit factor 1.99; win rate 71.2%.

Transaction cost is deducted once as a 0.30% round-trip charge, and short borrow is deducted once as 0.50% for the modeled five-day hold. Notional is fixed at initial capital divided by maximum positions, so scenario sizing is comparable. Same-symbol overlaps are rejected; the unfiltered run rejected 0 such candidates.

## Concentration and extreme-trade dependence

The unfiltered short's largest profitable trade accounts for 3.6% of net P&L and its five largest account for 16.1%. Under the $5/$1M filter those shares are 3.9% and 17.4%. Those shares do not indicate dependence on one or a handful of extreme winners.

Microcap and difficult-to-borrow concentration cannot be measured directly: market capitalization is available=False, and historical borrow availability is available=False. Price and dollar-volume filters are only proxies. Any claim that this strategy avoids microcaps or hard-to-borrow names would be unsupported.

## Leakage and data quality

- No forward outcome column or next-session entry price is selected as a feature.
- Chronological train/validation/test periods are disjoint, and labels crossing the 2019 or 2022 boundary are purged. The audit found and removed boundary-crossing candidates rather than silently training on them: {'2019-12-31': 5, '2022-12-31': 3}.
- Every walk-forward fold trains only on observations whose complete label ends before January 1 of the test year.
- Portfolio thresholds are computed only from validation predictions; test predictions never set the cutoff.
- Feature/tradeability data is joined with a validated many-to-one merge on symbol plus normalized event date. Duplicate keys: 0; missing raw price dates: 0; event-close mismatches against raw symbol files: 0 of 2198 checked.

There is no direct leakage found after the boundary purge. Overfitting and research-selection bias remain plausible if the test years or filter scenarios have already influenced feature/model choices. The untouched-test concept has weakened through repeated inspection, so these results should not be treated as a pristine final holdout.

## Bottom line

The short edge survives both tested price/liquidity filters: even the strict scenario retains positive CAGR and a profit factor above one. The strict scenario is the more credible headline number. Even then, paper trading requires historical locate/borrow data, daily mark-to-market and halt modeling, and a genuinely untouched forward period. Until those gaps are closed, the strategy is research-only.

## Subsequent daily mark-to-market research

Daily path reconstruction confirms that the earlier realized-equity statistics understated short-squeeze risk. Under a 70% locate scenario, 10% annual borrow, 1% round-trip cost, and the risk rule selected only on 2020-2022 validation data, the 2023+ test produced 15.68% CAGR, -20.55% maximum drawdown, 1.03 daily Sharpe, and 2.13 profit factor across 47 located trades. Aligned SPY returned 71.52% versus the strategy's 59.69%.

The worst short lost 322.1%, and all conventional 20%-100% stops lost money during validation. This is now the controlling conclusion: the ranking signal is real enough to continue studying, but no historically validated exit rule safely contains squeeze risk. The strategy remains unsuitable for paper trading until broker margin, forced liquidation, actual locates, and quoted borrow rates are modeled prospectively. See `reports/v5_mtm_research/ASSESSMENT.md` for the full stress analysis.

Subsequent gross-capped S&P and two-sided V5 overlay tests also failed: every nonzero allocation reduced validation performance versus SPY, so validation selected a zero allocation. The generalized four-family Alpha Factory likewise missed its target with 8.14% CAGR, 24.79% drawdown, and 0.54 Sharpe. See `reports/alternative_research/FINAL_DECISION.md` for the consolidated rejection decision.
