# Nasdaq ITCH Order-Imbalance Sample Assessment

## Verdict

The public Nasdaq sample successfully validates order-book reconstruction and the
quote-aware execution pipeline. The tested imbalance/Markov-persistence signal has no
edge: every shortlisted validation result is negative. It is rejected.

This is one MSFT session from January 3, 2003. It cannot support an annual-return claim,
modern-market inference, or strategy promotion regardless of its result. The final
intraday partition was deliberately withheld.

## Data and reconstruction

- Official public Nasdaq ITCH 2.0 sample: `S010303-v2.zip`.
- Raw exchange messages: 6,115,981.
- Visible add orders: 2,921,796.
- Cancels: 2,762,898.
- Visible executions: 257,338.
- Hidden/non-displayed trade messages: 173,947.
- Reconstructed MSFT regular-session top-of-book changes: 31,796.
- Median displayed spread: 6.92 bps.
- 95th-percentile spread: 10.96 bps.
- Median displayed size: 500 shares on each side.

ITCH event sequence is used directly. Multiple messages sharing a millisecond are
processed in feed order, and only the final changed book at that timestamp is exported.

## Signal protocol

Twenty-seven fixed candidates combine:

- displayed-size imbalance thresholds of 0.20, 0.40, and 0.60;
- online first-order state-persistence thresholds of 0.40, 0.60, and 0.80; and
- holding times of 1, 5, and 30 seconds.

The transition matrix is updated online with Laplace smoothing. Signals use only the
current and prior book. Orders arrive 100 ms later, may consume no more than 10% of the
displayed touch, pay the spread plus square-root impact, and pay a $0.005/share commission
with a $0.35 minimum per side. Each position uses at most 10% of the $10,000 account.

The first half-session ranks candidates. Only the top five are evaluated on the next
quarter-session. The last quarter-session is not evaluated because one historical day
cannot produce a statistically credible lock.

## Results

- Validation-positive shortlisted candidates: **0 of 5**.
- Eligible candidates: **0**.
- Best validation result: **-$28.22**, or **-28.22 bps** on account equity.
- Best candidate validation trades: 19.
- Best candidate validation win rate: 0%.
- Best candidate validation profit factor: 0.
- Locked final partition evaluated: **no**.

The highest-turnover one-second candidate lost $3,683.01 during the training half-session
and generated thousands of depth rejections. Both adverse selection and explicit costs
are material; this is not merely a commission problem.

## Conclusion

Naive displayed imbalance, even conditioned on online Markov persistence, is not a
tradeable edge in this sample. The infrastructure now accepts raw exchange events and
produces realistic top-of-book fills, but multi-year modern quote/trade data is still
required for genuine research. The 50% CAGR objective remains unmet and allocation stays
at zero.
