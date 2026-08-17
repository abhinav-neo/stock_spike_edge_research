# Broker Margin and Forced-Liquidation Assessment

## Verdict

**REJECT.** No margin profile is promoted unless it reaches the locked 40% CAGR
target, keeps drawdown within 25%, bounds every loss to at most the allocated notional,
and beats aligned SPY. Margin rules are specified independently of test returns.

## Results

| Profile | Liquidations | CAGR | Total return | Max drawdown | Sharpe | Worst trade |
|---|---:|---:|---:|---:|---:|---:|
| unconstrained | 0 | 15.68% | 59.69% | -20.55% | 1.03 | -322.1% |
| reg_t_50_30 | 32 | -2.39% | -7.43% | -24.85% | -0.13 | -46.9% |
| house_100_50 | 20 | 6.57% | 22.57% | -21.42% | 0.52 | -46.9% |
| hard_to_borrow_200_100 | 11 | 11.64% | 42.46% | -13.71% | 0.84 | -51.4% |

The profiles model position-level collateral: Reg T-like 50% initial/30% maintenance,
a 100%/50% house rule, and a 200%/100% hard-to-borrow rule. Gap-through liquidations
fill at the next available open. These scenarios still cannot reproduce discretionary
broker recalls, symbol-specific house-margin changes, or account-wide cross-margining.
