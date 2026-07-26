# U.S. Stock +40% Daily Spike Event Study

This project identifies U.S. stocks that gained at least 40% from one adjusted
daily close to the next, measures how long the spike was retained, and searches
for continuation and failed-spike mean-reversion edges.

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

The broad candidate search is intentionally simple and interpretable. Do not
optimize thousands of opaque model parameters before establishing whether the
basic phenomenon is stable.

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
