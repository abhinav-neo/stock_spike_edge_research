# Point-in-Time Intraday Feature Assessment

## Verdict

**REJECT.** SIP minute bars were collected only through the 2022 validation boundary;
test intraday data and returns were not used for selection. Validation coverage with at
least 30 bars was 78.3%. The best variant was `path` with
hist_gradient_boosting, improving validation correlation by 0.0059 versus
the locked +0.0200 requirement.

Allocation remains zero unless a validation-eligible variant is locked before test data
is collected.
