from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

from src.point_in_time_data import asof_join_events
from src.v5_mtm_research import attach_event_dates, build_period_trades


FINRA_URL = "https://api.finra.org/data/group/otcMarket/name/consolidatedShortInterest"
PUBLICATION_LAG_DAYS = 14


class FinraShortInterestClient:
    def __init__(self, session: requests.Session | None = None, retries: int = 5) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})
        self.retries = retries

    def history(self, symbol: str) -> list[dict]:
        payload = {
            "limit": 5000,
            "compareFilters": [
                {"compareType": "EQUAL", "fieldName": "symbolCode", "fieldValue": symbol}
            ],
        }
        for attempt in range(self.retries):
            response = self.session.post(FINRA_URL, json=payload, timeout=60)
            if response.status_code == 429 or response.status_code >= 500:
                if attempt + 1 == self.retries:
                    response.raise_for_status()
                time.sleep(min(2**attempt, 20))
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("FINRA short-interest request exhausted retries")


def normalize_short_interest(records: list[dict]) -> pd.DataFrame:
    columns = [
        "symbol", "asof_date", "settlement_date", "short_interest",
        "short_interest_previous", "average_daily_volume", "days_to_cover",
        "short_interest_to_adv", "stock_split_flag",
    ]
    if not records:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(records).rename(
        columns={
            "symbolCode": "symbol",
            "settlementDate": "settlement_date",
            "currentShortPositionQuantity": "short_interest",
            "previousShortPositionQuantity": "short_interest_previous",
            "averageDailyVolumeQuantity": "average_daily_volume",
            "daysToCoverQuantity": "days_to_cover",
            "stockSplitFlag": "stock_split_flag",
        }
    )
    frame["settlement_date"] = pd.to_datetime(frame["settlement_date"], errors="coerce").dt.normalize()
    frame["asof_date"] = frame["settlement_date"] + pd.Timedelta(days=PUBLICATION_LAG_DAYS)
    numeric = ["short_interest", "short_interest_previous", "average_daily_volume", "days_to_cover"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    denominator = frame["average_daily_volume"].where(frame["average_daily_volume"].gt(0))
    frame["short_interest_to_adv"] = frame["short_interest"] / denominator
    if "stock_split_flag" not in frame:
        frame["stock_split_flag"] = pd.NA
    frame = frame.loc[frame["symbol"].notna() & frame["settlement_date"].notna()].copy()
    return frame.sort_values(["symbol", "asof_date", "settlement_date"]).drop_duplicates(
        ["symbol", "asof_date"], keep="last"
    )[columns].reset_index(drop=True)


def attach_short_interest(events: pd.DataFrame, short_interest: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["event_date"] = pd.to_datetime(events["event_date"]).astype("datetime64[ns]")
    external = short_interest.drop(columns="settlement_date", errors="ignore")
    external = external.copy()
    external["asof_date"] = pd.to_datetime(external["asof_date"]).astype("datetime64[ns]")
    joined = asof_join_events(events, external, max_staleness_days=45)
    joined = joined.rename(
        columns={
            "pit_asof_date": "short_interest_available_date",
            "pit_staleness_days": "short_interest_staleness_days",
            "pit_is_stale": "short_interest_is_stale",
        }
    )
    joined["short_interest_data_available"] = joined["short_interest_available_date"].notna()
    return joined


def write_assessment(joined: pd.DataFrame, raw: pd.DataFrame) -> str:
    counts = joined.groupby("period").agg(
        candidates=("symbol", "size"),
        covered=("short_interest_data_available", "sum"),
    )
    rows = ["| Period | Candidates | Covered | Coverage |", "|---|---:|---:|---:|"]
    for period, row in counts.iterrows():
        coverage = row.covered / row.candidates if row.candidates else 0
        rows.append(f"| {period} | {int(row.candidates)} | {int(row.covered)} | {coverage:.1%} |")
    validation_covered = int(counts.loc["validation", "covered"]) if "validation" in counts.index else 0
    return f"""# FINRA Consolidated Short-Interest Coverage

## Verdict

**Not eligible for historical model promotion.** Only {validation_covered} validation
candidates have usable point-in-time observations, below the locked minimum of 30.
No short-interest threshold was selected and test-period returns were not evaluated.

## Coverage

FINRA returned {len(raw):,} semi-monthly symbol records. To prevent publication
lookahead, each settlement is made feature-eligible only after a conservative
{PUBLICATION_LAG_DAYS}-calendar-day lag; observations older than 45 days are rejected.

{chr(10).join(rows)}

The collector remains available for prospective coverage. Any later model use must
first pass the locked validation-improvement and sample-size gates, without choosing
thresholds from test outcomes. Allocation remains zero.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect point-in-time FINRA consolidated short interest")
    parser.add_argument("--predictions", default="reports/v5_validation_suite/model/predictions.csv")
    parser.add_argument("--features", default="data/processed/events_features_v5.parquet")
    parser.add_argument("--output-dir", default="reports/finra_short_interest")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    features = pd.read_parquet(args.features)
    periods = []
    for period in ("validation", "test"):
        frame = attach_event_dates(build_period_trades(predictions, features, period), features, 5)
        frame["period"] = period
        periods.append(frame)
    candidates = pd.concat(periods, ignore_index=True)

    client = FinraShortInterestClient()
    records = []
    for symbol in sorted(candidates["symbol"].astype(str).unique()):
        records.extend(client.history(symbol))
    raw = normalize_short_interest(records)
    joined = attach_short_interest(candidates, raw)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output / "short_interest_point_in_time.csv", index=False)
    joined.to_csv(output / "candidate_short_interest.csv", index=False)
    summary = {
        "records": len(raw),
        "candidates": len(joined),
        "covered": int(joined["short_interest_data_available"].sum()),
        "publication_lag_days": PUBLICATION_LAG_DAYS,
        "maximum_staleness_days": 45,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "ASSESSMENT.md").write_text(write_assessment(joined, raw), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
