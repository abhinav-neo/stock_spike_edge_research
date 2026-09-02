# U.S. Stock +40% Daily Spike Event Study

This project identifies U.S. stocks that gained at least 40% from one adjusted
daily close to the next, measures how long the spike was retained, and searches
for continuation and failed-spike mean-reversion edges.

## V2 methodology

Version 2 changes the implementation to better reflect tradable execution:

- Event detection still uses a +40% move from the prior close, but the event is
  evaluated with consistently adjusted OHLC price levels rather than mixing
  adjusted close with raw open/high/low values.
- The tradable entry uses the next trading day's opening price, not the
  event-day close.
- Forward returns are calculated from the next-day open for 1, 2, 3, 5, 10,
  20, 40, and 60 trading days. A 1-day return exits at the close of the entry
  session; an H-day return exits at the close of the Hth trading session after
  the event.
- A configurable per-symbol cooldown (default 20 calendar days) suppresses
  overlapping correlated events to reduce look-ahead and clustering effects.
- The pipeline retains chronological train/validation/test splits and reports
  sample size, mean, median, win rate, t-statistic, and robust score for each
  rule while preserving the existing rule naming convention.

## Important limitations

This is a zero-cost research pipeline. It is materially better than testing only
an S&P 500 list, but it is not identical to CRSP or Norgate:

- Yahoo/yfinance does not guarantee full coverage of every delisted ticker.
- Historical symbols may have changed or become unavailable.
- Free data can contain stale values and adjustment errors.
- Borrow availability, halt risk, bid/ask spreads and hard-to-borrow fees are
  not available historically.
- Candidate short edges must therefore be considered provisional until paper
  trading validates execution.

The code avoids claiming an edge unless it is positive in training, validation
and untouched test periods after modeled costs.

## Data sources

1. Nasdaq Trader symbol directories: current U.S.-listed securities.
2. Alpha Vantage LISTING_STATUS: optional active/delisted metadata from 2010 onward.
3. yfinance: adjusted and unadjusted OHLCV history.

## Installation on Windows

```powershell
cd stock_spike_edge_research
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional Alpha Vantage key:

```powershell
$env:ALPHA_VANTAGE_API_KEY="YOUR_FREE_KEY"
```

## First test: 200 symbols

This validates that downloads and calculations work before attempting the full universe.

```powershell
python -m src.universe
python -m src.download_prices --limit 200
python -m src.event_study
python -m src.analyze_edges
```

## Full run

```powershell
python -m src.run_all
```

Downloading thousands of symbols from a free unofficial endpoint can trigger
temporary throttling. The downloader resumes because it skips existing Parquet
files. Re-run the same command after a pause when required.

## Outputs

- `data/raw/universe.csv`: symbols and available listing metadata
- `data/raw/prices/*.parquet`: one daily-history file per symbol
- `data/processed/events.parquet`: one row per +40% event
- `reports/retention_summary.csv`: answer to how long prices retain the spike
- `reports/candidate_edges.csv`: all tested conditional rules
- `reports/accepted_edges.csv`: rules that remained positive in all periods and
  meet minimum sample-size requirements

## Core event definition

An event is:

```text
adjusted_close[t] / adjusted_close[t-1] - 1 >= 40%
```

In V2, the tradeable entry is implemented as:

```text
entry_price = next_trading_day_open
exit_price = entry_price * (1 + forward_return_h)
```

Filters:

- Previous close at least $1
- Prior 20-day average dollar volume at least $1 million
- Event-day dollar volume at least $5 million
- No stock split reported on the event day
- Implausible daily changes above 1,000% excluded by default

Edit `config/config.yaml` to change the assumptions.

## Validation design

- Training: through December 31, 2019
- Validation: January 1, 2020 through December 31, 2022
- Untouched test: January 1, 2023 onward
- Default round-trip friction: 100 basis points total
- Minimum samples: 100 training events and 30 events in each later period
- Cooldown: events for the same symbol occurring 20 or fewer calendar days
  after the prior accepted event are skipped by default; configurable with
  `research.cooldown_days` in `config/config.yaml`

The broad candidate search is intentionally simple and interpretable. Do not
optimize thousands of opaque model parameters before establishing whether the
basic phenomenon is stable.

## Commands

Run the full workflow with the V2 logic:

```powershell
python -m src.universe
python -m src.download_prices --limit 200
python -m src.event_study
python -m src.analyze_edges
```

To run the unit tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## Zero-capital forward observation

The historical candidate search is closed. The accepted alpha allocation is zero, and
the forward workflow records observations and execution feasibility without submitting
orders. Run it after the U.S. market close with Alpaca credentials available in the
process environment:

```powershell
.\.venv\Scripts\python.exe -B -m src.daily_forward_pipeline --end-date YYYY-MM-DD
```

For an unattended Windows run, invoke
`scripts\run_daily_forward_observation.ps1` from Task Scheduler after market close.
The script appends output to `reports\forward_observation\scheduled_run.log` and returns
a failing exit code when any pipeline stage fails. Re-running a date is safe: market
data is merged by symbol/date, signals are deduplicated by their locked identity, and
state files are replaced atomically.

The market-data updater scales by grouping the existing universe once, downloads stale
symbols in batches, retries failed batches by symbol, and bounds provider requests. It
also rolls an in-progress U.S. session back to the latest completed weekday, using an
18:00 America/New_York availability cutoff, so a premarket or intraday scheduled run
cannot ingest an incomplete daily bar. Exchange holidays may still appear in the
reported `no_data_symbols` list and are retried safely on the next run.

The workflow deliberately excludes `paper_trade_alpha`, `paper_fill_tracker`, and every
live order-submission path. Forward evidence cannot become a breakthrough until both
the statistical and operational gates in `config/alpha_factory.yaml` pass.

Actual broker locate decisions are intentionally separate from Alpaca's general
`shortable` and `easy_to_borrow` asset flags. Provider exports can be validated and
appended with `python -m src.forward_locate_evidence --input <provider-export.csv>`.
Each row must identify the locked observation, decision timestamp, provider, request
and confirmation flags, quoted annual borrow rate, available quantity, and a redacted
source reference. Missing, duplicate, or unknown decisions cannot satisfy the gate.

For Alpaca ETB securities, the daily pipeline records the broker-established locate
basis and zero borrow rate from the current `borrow_status` field. It never submits an
HTB locate request. HTB signals remain quarantined unless a separately validated
explicit locate record is supplied.

## Limitations

- The next-open entry assumption still ignores market impact, spreads, slippage,
  and partial fills that materially matter for real execution.
- Adjusted prices are based on the available OHLCV series and may still be
  imperfect when corporate actions or data vendor adjustments are inconsistent.
- Cooldown filtering reduces overlap but does not fully remove clustering or
  correlated event contamination.
- The project remains a research pipeline and should not be treated as a
  production-ready trading system.

## How to interpret accepted_edges.csv

A row is not automatically ready for live trading. Before deployment:

1. Inspect yearly returns rather than only the pooled average.
2. Remove acquisition announcements from ordinary momentum events.
3. Paper-trade using executable bid/ask prices.
4. For shorts, log actual locate availability and borrow fees.
5. Set portfolio-level exposure and daily-loss limits.
6. Reject an edge that depends on a few extreme trades.

## Suggested first live strategy candidate

Do not select it until the report supports it. The initial hypotheses are:

- **Long continuation:** event closes near the session high, high relative
  dollar volume, liquid stock and next-day confirmation.
- **Failed-spike short:** event closes in the lower half of its range and fails
  the event midpoint, subject to actual borrow availability.
- **Long first pullback:** strong event followed by a controlled pullback that
  remains above the pre-event close.

The statistically strongest and operationally feasible rule—not the most
profitable in-sample rule—should be selected.
