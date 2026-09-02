# Free FINRA Short-Volume Research Assessment

## Verdict

The free FINRA signal is retained as a research feature but does not make the strategy acceptable for paper trading. The ratio-only specification provides a small, mixed improvement; the absolute-volume specification is rejected.

## Data and semantics

- Source: FINRA Daily Short Sale Volume, free for non-commercial use under FINRA's terms.
- Coverage: 1,835 of 2,198 V5 events (83.5%) across 1,186 requested dates.
- Historical files combine FINRA/Nasdaq and FINRA/NYSE facilities; later dates use FINRA's consolidated NMS file.
- The join is exact on normalized symbol and event date. Missing observations remain missing and receive a separate availability flag.
- Event-day data is used only for a next-session entry. It is never joined to an earlier event.
- This is off-exchange short-sale transaction volume. It is not short interest, borrow cost, locate availability, utilization, or evidence that a short remained open.

## Model comparison

| Specification | Validation correlation | Test correlation | Walk-forward mean correlation | Walk-forward mean spread |
|---|---:|---:|---:|---:|
| V5 baseline | 0.0661 | 0.1523 | 0.1534 | 25.40% |
| FINRA all fields | 0.0587 | 0.1602 | 0.1501 | 18.10% |
| FINRA ratio only | 0.0700 | 0.1489 | 0.1611 | 24.56% |

The all-field version deteriorates on validation and walk-forward tests. The ratio-only version improves validation and mean yearly correlation slightly, but worsens the test correlation and remains below V5 on yearly top-bottom spread. Its FINRA ratio permutation importance is positive in all seven yearly folds but small in magnitude.

## Strict tradeable portfolio

The ratio-only model, with a validation-calibrated bottom-decile threshold, minimum $10 entry price, minimum $5 million prior 20-day average dollar volume, five-day holding period, 30 bps round-trip cost, and 10 bps daily borrow assumption produced:

- 75 trades;
- 22.27% CAGR and 93.72% total return;
- 17.91% realized-equity maximum drawdown;
- 1.67 trade-level Sharpe proxy;
- 2.05 profit factor and 72.0% win rate;
- 4.53% approximate capital utilization;
- 9.66% of P&L from the largest winner and 40.06% from the five largest winners; and
- a catastrophic worst trade of -321.70% on allocated short notional.

The corresponding V5 strict scenario had 73 trades, 21.11% CAGR, 18.46% drawdown, 1.59 Sharpe proxy, and 1.99 profit factor. The FINRA improvement is modest and does not repair the core execution gap. A short can lose more than its initial notional, and the fixed-horizon realized-equity summary understates the margin and liquidation consequences of that path.

## Controlling conclusion

Keep `finra_short_volume_ratio` available for future research, but allocate no capital. It is a weak incremental feature, not a free substitute for borrow or locate history. The strategy remains unsuitable for paper trading because the free dataset cannot establish executability and the observed catastrophic short tail is incompatible with the requested robustness standard.
