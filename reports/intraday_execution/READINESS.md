# Intraday Research and Execution Readiness

## Current status

The repository now has a quote-level execution kernel for a $10,000 account, but it
does not contain intraday quotes/trades and no market-data credential is configured.
Consequently, no high-frequency return, CAGR, drawdown, or trade-frequency claim can
yet be measured honestly.

## Required point-in-time inputs

At minimum, each historical quote must contain:

- symbol and UTC timestamp with sub-second precision;
- national best bid and offer prices;
- displayed bid and ask sizes;
- exchange/session condition or a pre-cleaned regular-session guarantee;
- split, symbol-change, delisting, and trading-halt history; and
- a survivorship-free universe definition available as of each historical date.

Trades should additionally contain price, size, exchange, and sale condition. Sequence
numbers are strongly preferred so replay order is deterministic. One-minute OHLCV bars
are insufficient for validating spread capture, queue position, latency, or marketable
fill quality.

## Implemented realism controls

`src/intraday_execution.py` currently enforces:

- fills only at or after signal time plus configured latency;
- buys at the ask and sells at the bid before additional impact;
- square-root impact as participation at displayed touch increases;
- a maximum fraction of displayed touch size;
- per-share/minimum commissions and sale-side regulatory fees;
- rejection when no post-arrival quote or adequate displayed depth exists;
- crossed/invalid/duplicate quote rejection; and
- equity, CAGR, drawdown, trades/day, win rate, and gross-to-net P&L from $10,000.

This is intentionally conservative but not yet a complete limit-order queue simulator.
It must not be used to claim executable performance without real quote data.

## Account constraints to lock before promotion

The live broker/account type must be specified. A U.S. equity margin account may impose
pattern-day-trading equity requirements, while a cash account cannot freely recycle
unsettled sale proceeds. The production gate must query and enforce the broker's actual
buying power, settled cash, short availability, and day-trade status rather than assume
that the full $10,000 can be reused for every intraday signal.

## Unblocking artifact

Provide a historical quote/trade dataset as partitioned Parquet, or configure credentials
for a provider that permits historical research and automated use. A useful initial study
needs at least two years spanning different volatility regimes, including delisted names;
five or more years is preferable for the requested 50% CAGR and drawdown claim.
