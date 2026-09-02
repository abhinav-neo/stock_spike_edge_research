from __future__ import annotations

import argparse
import html
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from src.alpaca_historical_data import credentials_from_environment
from src.v5_mtm_research import attach_event_dates, build_period_trades


ALPACA_CORPORATE_ACTIONS_URL = "https://data.alpaca.markets/v1/corporate-actions"
NASDAQ_HALTS_URL = "https://www.nasdaqtrader.com/rss.aspx"


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


class AlpacaCorporateActionsClient:
    def __init__(self, key: str, secret: str, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    def actions(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        rows: list[dict] = []
        for symbol_chunk in _chunks(sorted(set(symbols)), 100):
            params = {"symbols": ",".join(symbol_chunk), "start": start, "end": end, "limit": 1000}
            while True:
                response = self.session.get(ALPACA_CORPORATE_ACTIONS_URL, params=params, timeout=60)
                response.raise_for_status()
                payload = response.json()
                for action_type, actions in (payload.get("corporate_actions") or {}).items():
                    for action in actions:
                        rows.append({"action_type": action_type, **action})
                token = payload.get("next_page_token")
                if not token:
                    break
                params["page_token"] = token
        return normalize_actions(pd.DataFrame(rows))


def normalize_actions(actions: pd.DataFrame) -> pd.DataFrame:
    columns = ["action_type", "symbol", "effective_date", "id"]
    if actions.empty:
        return pd.DataFrame(columns=columns)
    result = actions.copy()
    symbol = result.get("symbol", pd.Series(index=result.index, dtype="object"))
    if "old_symbol" in result:
        symbol = symbol.fillna(result["old_symbol"])
    result["symbol"] = symbol.astype("string")
    process_date = result.get("process_date", pd.Series(index=result.index, dtype="object"))
    ex_date = result.get("ex_date", pd.Series(index=result.index, dtype="object"))
    result["effective_date"] = pd.to_datetime(process_date.fillna(ex_date), errors="coerce").dt.normalize()
    return result.loc[result["symbol"].notna() & result["effective_date"].notna()].reset_index(drop=True)


def _rss_value(item: str, field: str) -> str:
    match = re.search(
        rf"<(?:[A-Za-z0-9_-]+:)?{field}(?:\s[^>]*)?>(.*?)</(?:[A-Za-z0-9_-]+:)?{field}>",
        item,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return html.unescape(re.sub(r"<[^>]+>", "", match.group(1))).strip() if match else ""


def parse_nasdaq_halt_rss(content: bytes) -> pd.DataFrame:
    text = content.decode("utf-8-sig", errors="replace")
    items = re.findall(r"<item(?:\s[^>]*)?>(.*?)</item>", text, flags=re.IGNORECASE | re.DOTALL)
    rows = []
    for item in items:
        rows.append(
            {
                "symbol": _rss_value(item, "IssueSymbol"),
                "issue_name": _rss_value(item, "IssueName"),
                "market": _rss_value(item, "Mkt"),
                "reason_code": _rss_value(item, "ReasonCode"),
                "halt_date": _rss_value(item, "HaltDate"),
                "halt_time": _rss_value(item, "HaltTime"),
                "resumption_date": _rss_value(item, "ResumptionDate"),
                "resumption_trade_time": _rss_value(item, "ResumptionTradeTime"),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["symbol", "halt_date", "halt_time", "reason_code"])
    result["halt_date"] = pd.to_datetime(result["halt_date"], errors="coerce").dt.normalize()
    return result.loc[result["symbol"].ne("") & result["halt_date"].notna()].reset_index(drop=True)


class NasdaqHaltClient:
    def __init__(self, session: requests.Session | None = None, retries: int = 3) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": "stock-spike-edge-research/1.0"})
        self.retries = retries

    def halts(self, date: pd.Timestamp) -> pd.DataFrame:
        for attempt in range(self.retries):
            response = self.session.get(
                NASDAQ_HALTS_URL,
                params={"feed": "tradehalts", "haltdate": pd.Timestamp(date).strftime("%m%d%Y")},
                timeout=60,
            )
            if response.status_code == 200:
                return parse_nasdaq_halt_rss(response.content)
            if attempt + 1 == self.retries:
                response.raise_for_status()
            time.sleep(2**attempt)
        raise RuntimeError("Nasdaq halt request exhausted retries")


def annotate_candidates(candidates: pd.DataFrame, actions: pd.DataFrame, halts: pd.DataFrame) -> pd.DataFrame:
    result = candidates.copy()
    result["event_date"] = pd.to_datetime(result["event_date"]).dt.tz_localize(None).dt.normalize()
    reverse_splits = actions.loc[actions["action_type"].eq("reverse_splits")]
    for days in (30, 90, 180, 365):
        flags = []
        for row in result.itertuples(index=False):
            age = row.event_date - reverse_splits["effective_date"]
            mask = (reverse_splits["symbol"] == row.symbol) & age.between(pd.Timedelta(0), pd.Timedelta(days=days))
            flags.append(bool(mask.any()))
        result[f"reverse_split_within_{days}d"] = flags
    halt_dates = pd.to_datetime(halts.get("halt_date", pd.Series(dtype="datetime64[ns]")))
    halt_keys = set(zip(halts.get("symbol", pd.Series(dtype="object")), halt_dates))
    result["event_day_halt"] = [(row.symbol, row.event_date) in halt_keys for row in result.itertuples(index=False)]
    return result


def assessment(annotated: pd.DataFrame, actions: pd.DataFrame, halts: pd.DataFrame) -> str:
    counts = annotated.groupby("period").agg(
        candidates=("symbol", "size"),
        event_day_halts=("event_day_halt", "sum"),
        reverse_split_90d=("reverse_split_within_90d", "sum"),
        reverse_split_365d=("reverse_split_within_365d", "sum"),
    )
    rows = ["| Period | Candidates | Event-day halts | Reverse split <=90d | Reverse split <=365d |", "|---|---:|---:|---:|---:|"]
    for period, row in counts.iterrows():
        rows.append(f"| {period} | {int(row.candidates)} | {int(row.event_day_halts)} | {int(row.reverse_split_90d)} | {int(row.reverse_split_365d)} |")
    table = "\n".join(rows)
    return f"""# Corporate-Action and Trading-Halt Coverage Assessment

## Verdict

**Useful risk evidence, but not eligible for historical model promotion.** The official
sources returned {len(actions):,} corporate actions and {len(halts):,} halts across the
locked validation/test event dates. The candidate-level coverage is:

{table}

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
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure corporate-action and halt coverage")
    parser.add_argument("--predictions", default="reports/v5_validation_suite/model/predictions.csv")
    parser.add_argument("--features", default="data/processed/events_features_v5.parquet")
    parser.add_argument("--output-dir", default="reports/event_risk_coverage")
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions)
    features = pd.read_parquet(args.features)
    periods = []
    for period in ("validation", "test"):
        frame = attach_event_dates(build_period_trades(predictions, features, period), features, 5)
        frame["period"] = period
        periods.append(frame)
    candidates = pd.concat(periods, ignore_index=True)

    key, secret = credentials_from_environment()
    actions = AlpacaCorporateActionsClient(key, secret).actions(
        candidates["symbol"].astype(str).tolist(),
        str(pd.to_datetime(candidates["event_date"]).min().date()),
        str(pd.to_datetime(candidates["event_date"]).max().date()),
    )
    halt_client = NasdaqHaltClient()
    dates = sorted(pd.to_datetime(candidates["event_date"]).dt.normalize().unique())
    halt_frames = [halt_client.halts(date) for date in dates]
    halts = pd.concat(halt_frames, ignore_index=True) if halt_frames else pd.DataFrame()
    annotated = annotate_candidates(candidates, actions, halts)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    actions.to_csv(output / "corporate_actions.csv", index=False)
    halts.to_csv(output / "event_date_halts.csv", index=False)
    annotated.to_csv(output / "candidate_event_risk.csv", index=False)
    summary = {
        "candidates": len(annotated),
        "corporate_actions": len(actions),
        "event_date_halts": len(halts),
        "candidate_event_day_halts": int(annotated["event_day_halt"].sum()),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "ASSESSMENT.md").write_text(assessment(annotated, actions, halts), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
