# Forward Observation Status

## Operational result — 2026-08-17

The unattended entry point `scripts/run_daily_forward_observation.ps1` completed
successfully. It updated the full 2,330-symbol universe through the latest completed
session, regenerated the locked forward ledger, captured read-only Alpaca asset and
account evidence, reused or acquired eligible quote windows, evaluated executable
outcomes, and regenerated the combined verdict. No order was submitted.

The Windows task `StockSpikeForwardObservation` is enabled with weekday triggers at
18:30, 20:30, and 22:30 America/Chicago. Its most recent scheduled result was successful
and its next retry is 2026-08-17 at 22:30 America/Chicago; the later triggers provide
same-evening retries while all pipeline stages remain idempotent.

- Latest completed data session: 2026-08-17
- Locked observations: 26
- Open observations: 26
- Settled observations: 0
- Broker-eligible observations: 19
- Quote-window coverage rows: 18
- Account controls ready: yes
- Actual locate decisions required: 19
- Broker-established ETB locates confirmed: 19
- Event-risk evidence captured: 26
- Event-risk captures before entry: 2
- Estimated first open-observation settlement: 2026-08-24
- Estimated latest current-observation settlement: 2026-08-31
- Integrity gate: passed
- Statistical gate: not yet passed
- Operational gate: not yet passed
- Allocation fraction: 0%
- Verdict: `CONTINUE_FORWARD_COLLECTION`

The latest update completed through August 17 despite isolated no-data symbols, which
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
promotion. The two August 17 observations were captured before their inferred
next-session entry open and are the first causally eligible event-risk records.
