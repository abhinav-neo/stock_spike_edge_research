# Bounded-Loss Options Coverage Assessment

## Verdict

**Not eligible for historical promotion.** Alpaca historical options data begins in
February 2024. Of 73 locked test candidates, 58 occur inside
that window and 7 (12.1%) have at least one 14–45 DTE expiration
with two put strikes, the minimum contract topology for a vertical spread.

Contract topology alone is not execution evidence. The connected data API provides
historical trades and bars but no historical bid/ask quote endpoint; the free option
feed is indicative rather than actual OPRA. Therefore entry debit, exit credit, spread,
slippage, and fill feasibility cannot be reconstructed honestly. No return, CAGR, or
drawdown is calculated for this path, and test-period contract availability is not used
to tune the stock model.
