# Forward Observation Status

## Operational result — 2026-08-19

The unattended entry point `scripts/run_daily_forward_observation.ps1` completed
successfully. It updated the full 2,330-symbol universe through the latest completed
session, regenerated the locked forward ledger, captured read-only Alpaca asset and
account evidence, reused or acquired eligible quote windows, evaluated executable
outcomes, and regenerated the combined verdict. No order was submitted.

The Windows task `StockSpikeForwardObservation` is enabled with weekday triggers at
18:30, 20:30, and 22:30 America/Chicago. Its August 19 20:30 run completed successfully
with result code 0; the later trigger provides a same-evening retry while all pipeline
stages remain idempotent.

- Latest completed data session: 2026-08-19
- Locked observations: 29
- Open observations: 29
- Settled observations: 0
- Broker-eligible observations: 21
- Broker-rejected/quarantined observations: 8
- Quote-window coverage rows: 20
- Median captured touch spread: 1,606 bps (gate maximum: 50 bps)
- Account controls ready: yes
- Actual locate decisions required: 21
- Broker-established ETB locates confirmed: 21
- Event-risk evidence captured: 29
- Event-risk captures before entry: 5
- Estimated first open-observation settlement: 2026-08-24
- Estimated latest current-observation settlement: 2026-09-02
- Integrity gate: passed
- Statistical gate: not yet passed
- Operational gate: not yet passed
- Allocation fraction: 0%
- Verdict: `CONTINUE_FORWARD_COLLECTION`

The latest update completed through August 19 despite isolated no-data symbols, which
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
promotion. Five observations from August 17 onward were captured before their inferred
next-session entry open and are the first causally eligible event-risk records.

Current quote evidence is operationally unfavorable: the median captured touch spread is
roughly 1,606 bps across 20 entry windows, versus the locked maximum of 50 bps. This is an
early coverage result, not a final verdict, because no observation has reached its exit
window. The threshold will not be weakened.
