# Forward Observation Status

## Operational result — 2026-09-02

The unattended entry point `scripts/run_daily_forward_observation.ps1` completed
successfully. It updated the full 2,330-symbol universe through the latest completed
session, regenerated the locked forward ledger, captured read-only Alpaca asset and
account evidence, reused or acquired eligible quote windows, evaluated executable
outcomes, and regenerated the combined verdict. No order was submitted.

The run log records a successful September 2 collection through the latest completed
session. The Windows task is configured with weekday triggers at 18:30, 20:30, and
22:30 America/Chicago; the later triggers provide same-evening retries while all
pipeline stages remain idempotent.

- Latest completed data session: 2026-09-01
- Locked observations: 45
- Open observations: 17
- Settled ledger observations: 28
- Execution-evaluable settled observations: 20
- Broker-eligible observations: 30
- Broker-rejected/quarantined observations: 15
- Quote-window coverage rows: 50
- Median captured touch spread: 2,817 bps (gate maximum: 50 bps)
- Account controls ready: yes
- Actual locate decisions required: 30
- Broker-established ETB locates confirmed: 30
- Event-risk evidence captured: 45
- Event-risk captures before entry: 21
- Estimated first open-observation settlement: 2026-09-02
- Estimated latest current-observation settlement: 2026-09-14
- Integrity gate: passed
- Statistical gate: not yet passed
- Operational gate: not yet passed
- Allocation fraction: 0%
- Verdict: `CONTINUE_FORWARD_COLLECTION`

The latest update completed through September 1 despite isolated no-data symbols, which
remain visible for subsequent retry or universe review.

## Promotion requirements still collecting

Promotion is deliberately impossible until the locked evidence reaches all configured
thresholds: at least 100 settled observations, 60 independent signal dates, 180 calendar
days, the required positive-candidate fraction, at least 95% executable quote coverage,
broker metadata coverage, shortability/easy-to-borrow thresholds, acceptable touch
spreads, actual locate evidence, and healthy paper-account controls.

These are elapsed forward-evidence requirements, not missing software. They must not be
backfilled, weakened, or bypassed to manufacture a production approval.

The locate gate now consumes a dedicated validated provider ledger rather than an asset
snapshot field. `shortable` and `easy_to_borrow` do not count as actual locate evidence.
Alpaca's current `borrow_status=easy_to_borrow` does count as broker-established ETB
evidence; no hard-to-borrow locate request or fee was submitted.

The unattended pipeline also records official corporate-action and Nasdaq halt evidence
on first observation. The original 24 observations are explicitly non-causal because the
collector was deployed after their entries; they cannot be used for event-risk feature
promotion. Twenty-one observations from August 17 onward were captured before their
inferred next-session entry open and are causally eligible event-risk records.

Current quote evidence is operationally unfavorable: the median captured touch spread is
roughly 2,817 bps across the current executable evidence, versus the locked maximum of
50 bps. The 20 execution-evaluable settled observations have a mean net return of 1.97%,
but the 95% confidence interval is -1.73% to 5.66% and only 50% are positive, versus the
locked 75% minimum. This remains an interim result because the 100-observation,
60-independent-date, and 180-day minimums are unmet. The thresholds will not be weakened.
