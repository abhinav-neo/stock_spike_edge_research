# SEC Fails-to-Deliver Research Assessment

## Verdict

Reject SEC fails-to-deliver features as a V5 promotion. Retain the leakage-safe collector for future research, but do not allocate capital or enable paper trading.

## Data integrity

- All 274 required half-month SEC archives from December 2014 through April 2026 were downloaded and parsed.
- The history contains 811,064 records for the V5 symbol universe and covers 83.6% of event rows.
- Seven historical archives required exceptional official SEC paths; the collector supports current, legacy FOIA, node-distribution, and the October 2019 suffixed file.
- Some SEC issuer descriptions contain an unescaped pipe. The parser preserves those records by taking fixed leading fields and the final price field.
- First-half observations are usable only from the first day of the following month; second-half observations are usable only from day 16 of the following month. The join is backward from publication availability, never settlement date.
- FTD is a cumulative settlement balance and may arise from long or short sales. It is not short interest, borrow availability, or evidence of abusive shorting.

## Model evidence

| Specification | Validation correlation | Test correlation | Walk-forward mean correlation | Walk-forward mean spread |
|---|---:|---:|---:|---:|
| V5 baseline | 0.0661 | 0.1523 | 0.1534 | 25.40% |
| FINRA ratio only | 0.0700 | 0.1489 | 0.1611 | 24.56% |
| SEC FTD | 0.0628 | 0.1538 | 0.1566 | 22.45% |
| FINRA ratio + SEC FTD | 0.0659 | 0.1567 | 0.1583 | 24.19% |

SEC FTD alone worsens validation and yearly spread. The combined model improves test and average yearly correlation, but fails to improve validation and remains below V5 on yearly spread. Promoting it because of the test improvement would be test-set selection.

## Strict portfolio and tail risk

The SEC FTD model's strict test portfolio produced 71 trades, 21.83% CAGR, 18.92% realized-equity drawdown, 1.67 trade-level Sharpe proxy, and 2.06 profit factor. It retained the same catastrophic SMX short, losing 321.70% of allocated notional.

SMX's lagged FTD quantity and dollar burden were below validation-period top-decile exclusion thresholds. Therefore, an honest FTD stress filter would not remove the catastrophic loss. Designing a cutoff specifically to exclude SMX after observing the test outcome would be overfitting.

## Controlling conclusion

The free SEC data closes another information gap but does not solve historical locate feasibility, borrow cost, recalls, halts, margin liquidation, or catastrophic short-tail exposure. The accepted allocation remains zero.
