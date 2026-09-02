# FINRA Consolidated Short-Interest Coverage

## Verdict

**Not eligible for historical model promotion.** Only 20 validation candidates have
usable point-in-time observations, below the locked minimum of 30. No short-interest
threshold was selected and test-period returns were not evaluated.

## Coverage

FINRA returned 9,299 semi-monthly symbol records. To prevent publication
lookahead, each settlement is made feature-eligible only after a conservative
14-calendar-day lag; observations older than 45 days are rejected.

| Period | Candidates | Covered | Coverage |
|---|---:|---:|---:|
| test | 73 | 62 | 84.9% |
| validation | 31 | 20 | 64.5% |

The collector remains available for prospective coverage. Any later model use must
first pass the locked validation-improvement and sample-size gates, without choosing
thresholds from test outcomes. Allocation remains zero.
