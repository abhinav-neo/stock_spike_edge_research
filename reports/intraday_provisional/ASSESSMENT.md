# Provisional Intraday Regime Research

## Verdict

No candidate meets the locked 50% CAGR, 25% maximum-drawdown, and five-trade-leg-per-day
gates. No candidate was allowed into the final chronological test partition. Live and
paper alpha allocation remain zero.

These experiments use adjusted OHLCV bars from a no-key convenience source. They are
pipeline and hypothesis tests, not production evidence: bars do not contain bid/ask,
displayed depth, sequence numbers, halts, delistings, or historical universe membership.

## Protocol

The research evaluated 288 configurations across:

- cross-sectional reversal and momentum;
- long-only and dollar-neutral long/short books;
- 1, 3, and 6-bar signal lookbacks;
- 1, 3, and 6-bar holding periods;
- all, quiet, trending, and stressed causal market regimes; and
- one or two selected symbols per side.

Signals are formed at a bar close and executed from the following bar open. Returns exit
at a later open, preventing same-bar look-ahead. Causal regimes use only prior SPY returns
and volatility percentiles. Every rebalance pays a locked 10 bp round-trip cost.

The first 50% of trading dates is training. Only the 12 candidates ranked on training are
evaluated on the following 25% validation period. The final 25% remains untouched unless
a candidate passes all validation gates.

## Five-minute sample

- Coverage: 2026-05-13 through 2026-08-07, 60 trading days.
- Symbols: 10; normalized rows: 46,800.
- Best shortlisted validation CAGR: **-49.0%**.
- Best shortlisted validation total return: **-3.83%**.
- Eligible candidates: **0**.

At this horizon, the tested cross-sectional effects are overwhelmed by turnover and the
10 bp cost assumption. The final partition was not evaluated.

## Hourly sample

- Coverage: 2023-09-11 through 2026-08-07, 730 trading days.
- Symbols: 10; normalized rows: 50,737.
- Best validation candidate: one-bar momentum, six-bar holding, quiet regime, long-only.
- Training CAGR: **17.1%** with **-32.9%** maximum drawdown.
- Validation CAGR: **41.5%** with **-13.0%** maximum drawdown.
- Validation frequency: **1.07 trade legs/day**.
- Eligible candidates: **0**.

The candidate misses both the 50% return and five-trades/day gates, and its training
drawdown exceeds 25%. It was not evaluated on the locked final partition.

## Conclusion

Increasing turnover does not manufacture the requested edge: the five-minute strategies
lose heavily after costs, while the slower hourly candidate improves but remains below
the return and frequency requirements. Further parameter tuning on these same samples
would be specification mining. The next valid step remains quote/trade acquisition and
research over a survivorship-free universe with broker/account constraints enforced.
