# Forward Execution Protocol

## SIP v1 — effective 2026-09-02

This protocol was declared before the September 2, 2026 market open. It applies only
to observations whose entry date is September 2, 2026 or later.

- Feed: Alpaca SIP historical quotes
- Entry fill: adverse touch on the first valid SIP quote at or after 09:30 ET
- Exit fill: adverse touch on the last valid SIP quote at or before 16:00 ET
- Round-trip cost: 100 bps, applied once
- Quote root: `data/raw/forward_quotes_sip_v1`
- Promotion statistics: executable SIP quote-side net returns only
- Capital allocation: 0%
- Order submission: disabled

The earlier IEX captures remain immutable diagnostic evidence. They are excluded from
promotion because the first IEX quote at the opening boundary was not a reliable proxy
for the market-wide executable touch. For example, the same HZO opening window showed
2,829 bps on the first IEX quote and 26.9 bps on the first SIP quote. No historical SIP
quotes are backfilled into the promotion cohort.

Changing this protocol requires a separately named prospective cohort. It must never be
rewritten in place after observing outcomes.
