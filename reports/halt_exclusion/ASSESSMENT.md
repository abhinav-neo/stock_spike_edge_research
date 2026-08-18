# Event-Day Halt Exclusion Assessment

## Verdict

**Rejected on validation sample size; locked-test return evaluation was not
authorized.** The fixed rule excludes every candidate with an official Nasdaq halt on
the signal date. It removes 14 of
31 validation candidates and retains only
17, below the locked minimum of
30.

The rule is operationally motivated and binary, but evaluating its test-period CAGR
after failing the validation gate would turn the test set into a selection tool. Halt
evidence remains mandatory prospective execution-risk telemetry. The rule may be
reconsidered only after a new forward sample supplies enough causally captured halted
and non-halted observations. Allocation and order submission remain disabled.
