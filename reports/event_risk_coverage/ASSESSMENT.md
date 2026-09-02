# Corporate-Action and Trading-Halt Coverage Assessment

## Verdict

**Useful risk evidence, but not eligible for historical model promotion.** The official
sources returned 182 corporate actions and 5,598 halts across the
locked validation/test event dates. The candidate-level coverage is:

| Period | Candidates | Event-day halts | Reverse split <=90d | Reverse split <=365d |
|---|---:|---:|---:|---:|
| test | 73 | 36 | 15 | 28 |
| validation | 31 | 14 | 1 | 1 |

The corporate-action endpoint exposes effective/process dates, not a guaranteed
point-in-time creation timestamp. Alpaca explicitly warns that action creation can be
delayed. A historical query therefore cannot prove that an action was observable to
the strategy on that date. In addition, validation has too few recent reverse-split
examples to select a stable exclusion window. Choosing a window after inspecting the
known test failures would be leakage.

Event-day halt matches are retained as execution-risk evidence, not as a return-tuned
filter. The forward collector must timestamp future corporate-action mutations and
halt observations prospectively before either feature can become promotion-eligible.
No allocation or trading state changes as a result of this study.
