# V6 Improvement Research Assessment

## Verdict

The freely available, leakage-safe improvement paths do not produce an acceptable strategy. Paper trading remains disabled.

## Market-regime context

Ten point-in-time SPY features were added using information available through each event close. Random-forest test correlation increased slightly from 0.1523 to 0.1554, but validation correlation fell from 0.0661 to 0.0625. More importantly, mean walk-forward correlation fell from 0.1534 to 0.1516 and mean top-bottom spread fell from 25.4% to 22.3%. The variant is rejected.

## Sector-relative context

Sector exposure was inferred from each stock's prior 60-session return correlation to 11 liquid sector ETFs. The event-day spike was excluded from sector inference. Forty-nine events with insufficient history were retained with an explicit `UNKNOWN` category.

Random-forest validation correlation was 0.0618 and test correlation was 0.1527, both worse than the V5 baseline. The variant is rejected before portfolio promotion.

## VIX context

VIX level, short returns, distance from its 20-day average, and rolling percentile were added point in time. Random-forest validation correlation was 0.0640 and test correlation was 0.1478. The variant is rejected.

## Alternate model

Histogram gradient boosting severely overfit both new feature sets. Training correlations exceeded 0.98 while test correlations fell to 0.106 for SPY/VIX context and 0.094 for sector context. Both variants are rejected.

## External data readiness

A strict as-of data interface now supports future market-cap, float, short-interest, borrow, halt, or fundamental inputs. It:

- requires `symbol` and `asof_date` keys;
- joins only the latest observation available on or before each event;
- rejects duplicate point-in-time keys;
- records staleness; and
- can clear observations older than a configured maximum age.

No current snapshot was backfilled into history. Doing so would create look-ahead bias. Historical borrow, float, short-interest, halt, options, or fundamentals still require a reliable point-in-time vendor or prospective collection.

## Controlling conclusion

All defensible free-data options are exhausted: richer technical features, SPY regime, VIX regime, inferred sector context, random forests, gradient boosting, stopped shorts, gross-capped overlays, two-sided portfolios, and four generalized alpha families. None passes the complete acceptance gate.

Further historical optimization on the existing panel is prohibited. The next improvement requires new point-in-time data or a new untouched forward period. Until then, the accepted alpha allocation is zero.
