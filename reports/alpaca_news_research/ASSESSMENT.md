# Point-in-Time News Feature Assessment

## Verdict

**REJECT.** News was collected only through the 2022 validation boundary. The test
news corpus and test returns were not used for selection. Validation event-news coverage
was 69.9%.

| Variant | Model | Validation correlation | Improvement | Eligible |
|---|---|---:|---:|---:|
| baseline | random_forest | 0.0661 | nan | False |
| topology | random_forest | 0.0668 | 0.0007 | False |
| all_news | random_forest | 0.0654 | -0.0007 | False |
| financing | random_forest | 0.0608 | -0.0053 | False |
| clinical | random_forest | 0.0612 | -0.0049 | False |
| merger | random_forest | 0.0609 | -0.0052 | False |
| contract | random_forest | 0.0621 | -0.0041 | False |
| earnings | random_forest | 0.0610 | -0.0051 | False |
| market_mechanics | random_forest | 0.0619 | -0.0042 | False |
| baseline | hist_gradient_boosting | 0.0651 | nan | False |
| topology | hist_gradient_boosting | 0.0625 | -0.0026 | False |
| all_news | hist_gradient_boosting | 0.0675 | 0.0024 | False |
| financing | hist_gradient_boosting | 0.0651 | 0.0000 | False |
| clinical | hist_gradient_boosting | 0.0656 | 0.0005 | False |
| merger | hist_gradient_boosting | 0.0651 | 0.0000 | False |
| contract | hist_gradient_boosting | 0.0651 | 0.0000 | False |
| earnings | hist_gradient_boosting | 0.0651 | 0.0000 | False |
| market_mechanics | hist_gradient_boosting | 0.0668 | 0.0017 | False |

Promotion requires at least +0.0200 absolute validation-correlation improvement without
increasing the train-validation gap by more than 0.05. Allocation remains zero.
