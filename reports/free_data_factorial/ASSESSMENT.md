# Free-Data Factorial Research Assessment

## Verdict

All available free-data feature combinations are rejected. No combination qualifies for locked test, portfolio, or paper-trading promotion.

## Protocol

Five independent feature groups were evaluated:

1. SPY market regime;
2. VIX regime;
3. inferred sector-relative context;
4. FINRA short-sale-volume ratio; and
5. publication-lagged SEC fails-to-deliver data.

The complete power set contains 32 combinations including the V5 baseline. Each was tested with random forest and histogram gradient boosting, producing 64 validation-only evaluations. Test outcomes were not used for ranking or locking.

A candidate had to improve random-forest validation correlation by at least 0.0200 and avoid adding more than 0.05 to the baseline train-validation correlation gap. These gates were fixed before inspecting the factorial results.

## Results

- V5 random-forest validation correlation: 0.0661.
- Best random forest: VIX + sector + FINRA + FTD at 0.0759, an improvement of only 0.0098.
- Best gradient boosting: VIX + sector + FINRA + FTD at 0.0916.
- Best gradient-boosting train correlation: 0.9845, creating a 0.8929 train-validation gap versus the baseline gap of 0.6077.
- Eligible candidates: 0 of 64.

The same four-group combination ranks first for both models, but neither result is acceptable. The random-forest gain is too small relative to 64-way validation search. Gradient boosting is severely overfit. Sector alone is the second-best random forest at 0.0727, which is also below the improvement gate.

## Multiplicity and test discipline

Selecting the maximum from 64 noisy validation results creates material multiple-comparison bias. A sub-threshold improvement cannot be treated as evidence merely because one combination ranks first. Since no candidate passed the validation gate, no new test-period model or portfolio was run. This avoids converting the repeatedly inspected test period into another tuning set.

## Controlling conclusion

The complete set of currently available free point-in-time and market-context combinations is exhausted. Further recombination, threshold adjustment, or model tuning on the same 2015-2026 observations would be specification mining. Accepted alpha allocation remains zero and paper trading remains rejected.
