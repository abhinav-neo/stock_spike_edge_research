from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

from src.alpaca_historical_data import credentials_from_environment
from src.free_data_factorial_research import evaluate_variant


NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
CATEGORY_PATTERNS = {
    "financing": re.compile(r"offering|registered direct|private placement|warrant|shelf|at-the-market|dilut", re.I),
    "clinical": re.compile(r"\bfda\b|clinical|phase [123]|trial|drug|therapy|patient|patent", re.I),
    "merger": re.compile(r"merger|acquisition|acquire|strategic alternative|takeover", re.I),
    "contract": re.compile(r"contract|purchase order|agreement|partnership|award", re.I),
    "earnings": re.compile(r"earnings|quarter|revenue|profit|loss|financial results", re.I),
    "market_mechanics": re.compile(r"halt|resume|circuit breaker|stocks moving|trading higher|trading lower", re.I),
}


class AlpacaNewsClient:
    def __init__(self, key: str, secret: str, session: requests.Session | None = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    def event_day(self, symbols: list[str], date: pd.Timestamp) -> list[dict]:
        day = pd.Timestamp(date).strftime("%Y-%m-%d")
        params = {
            "symbols": ",".join(sorted(set(symbols))),
            "start": f"{day}T00:00:00-04:00",
            "end": f"{day}T16:00:00-04:00",
            "limit": 50,
            "sort": "asc",
            "include_content": "false",
        }
        rows: list[dict] = []
        while True:
            response = self.session.get(NEWS_URL, params=params, timeout=60)
            if response.status_code == 429:
                time.sleep(float(response.headers.get("Retry-After", 3)))
                continue
            response.raise_for_status()
            payload = response.json()
            rows.extend(payload.get("news") or [])
            token = payload.get("next_page_token")
            if not token:
                return rows
            params["page_token"] = token


def article_text(article: dict) -> str:
    return f"{article.get('headline', '')} {article.get('summary', '')}".strip()


def event_news_features(events: pd.DataFrame, articles_by_date: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for event in events.itertuples(index=False):
        date = pd.Timestamp(event.event_date).normalize()
        articles = [
            item for item in articles_by_date.get(str(date.date()), [])
            if str(event.symbol) in (item.get("symbols") or [])
        ]
        specific = [item for item in articles if len(item.get("symbols") or []) <= 3]
        row = {
            "symbol": event.symbol,
            "event_date": date,
            "news_article_count": len(articles),
            "news_specific_article_count": len(specific),
            "news_has_specific_article": bool(specific),
            "news_specific_fraction": len(specific) / len(articles) if articles else 0.0,
            "news_min_tagged_symbols": min((len(item.get("symbols") or []) for item in articles), default=0),
        }
        for name, pattern in CATEGORY_PATTERNS.items():
            row[f"news_{name}_count"] = sum(bool(pattern.search(article_text(item))) for item in articles)
        rows.append(row)
    return pd.DataFrame(rows)


def collect_features(events: pd.DataFrame, client: AlpacaNewsClient, cache: Path) -> pd.DataFrame:
    cache.mkdir(parents=True, exist_ok=True)
    articles_by_date: dict[str, list[dict]] = {}
    grouped = events.assign(event_date=pd.to_datetime(events["event_date"]).dt.normalize()).groupby("event_date")
    for index, (date, group) in enumerate(grouped, start=1):
        key = str(date.date())
        target = cache / f"{key}.json"
        if target.exists():
            articles = json.loads(target.read_text(encoding="utf-8"))
        else:
            articles = client.event_day(group["symbol"].astype(str).tolist(), date)
            target.write_text(json.dumps(articles), encoding="utf-8")
            time.sleep(0.32)
        articles_by_date[key] = articles
        if index % 50 == 0 or index == len(grouped):
            print(f"News dates collected: {index}/{len(grouped)}", flush=True)
    return event_news_features(events, articles_by_date)


def validation_study(base: pd.DataFrame, news: pd.DataFrame, target: str, improvement: float) -> tuple[pd.DataFrame, dict]:
    keys = ["symbol", "event_date"]
    topology = [
        "news_article_count", "news_specific_article_count", "news_has_specific_article",
        "news_specific_fraction", "news_min_tagged_symbols",
    ]
    variants = {"topology": topology, "all_news": [column for column in news if column.startswith("news_")]}
    for name in CATEGORY_PATTERNS:
        variants[name] = [f"news_{name}_count"]
    rows = []
    for model in ("random_forest", "hist_gradient_boosting"):
        baseline, _ = evaluate_variant(base, (), model, target)
        baseline["eligible"] = False
        rows.append(baseline)
        for name, columns in variants.items():
            variant = base.merge(news[[*keys, *columns]], on=keys, how="left", validate="one_to_one")
            candidate, _ = evaluate_variant(variant, (name,), model, target)
            candidate["baseline_validation_correlation"] = baseline["validation_correlation"]
            candidate["absolute_improvement"] = candidate["validation_correlation"] - baseline["validation_correlation"]
            candidate["baseline_train_validation_gap"] = baseline["train_validation_gap"]
            candidate["eligible"] = bool(
                candidate["absolute_improvement"] >= improvement
                and candidate["train_validation_gap"] <= baseline["train_validation_gap"] + 0.05
            )
            rows.append(candidate)
    results = pd.DataFrame(rows)
    eligible = results.loc[results.get("eligible", False).fillna(False).astype(bool)]
    summary = {
        "protocol": "Train and validation news only; test news and returns remain unopened unless promoted.",
        "minimum_validation_correlation_improvement": improvement,
        "eligible_variants": int(len(eligible)),
        "test_news_collection_authorized": bool(len(eligible)),
    }
    return results, summary


def assessment(features: pd.DataFrame, results: pd.DataFrame, summary: dict) -> str:
    validation = features.loc[pd.to_datetime(features["event_date"]).between("2020-01-01", "2022-12-31")]
    coverage = float(validation["news_article_count"].gt(0).mean()) if len(validation) else 0.0
    table = "\n".join(
        ["| Variant | Model | Validation correlation | Improvement | Eligible |", "|---|---|---:|---:|---:|"]
        + [
            f"| {row.combination} | {row.model} | {row.validation_correlation:.4f} | "
            f"{getattr(row, 'absolute_improvement', float('nan')):.4f} | "
            f"{bool(getattr(row, 'eligible', False)) if pd.notna(getattr(row, 'eligible', False)) else False} |"
            for row in results.itertuples()
        ]
    )
    verdict = "PROMOTE TO LOCKED TEST COLLECTION" if summary["test_news_collection_authorized"] else "REJECT"
    return f"""# Point-in-Time News Feature Assessment

## Verdict

**{verdict}.** News was collected only through the 2022 validation boundary. The test
news corpus and test returns were not used for selection. Validation event-news coverage
was {coverage:.1%}.

{table}

Promotion requires at least +0.0200 absolute validation-correlation improvement without
increasing the train-validation gap by more than 0.05. Allocation remains zero.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only point-in-time Alpaca news study")
    parser.add_argument("--features", default="data/processed/events_features_v5.parquet")
    parser.add_argument("--through", default="2022-12-31")
    parser.add_argument("--target", default="forward_return_5d")
    parser.add_argument("--min-improvement", type=float, default=0.02)
    parser.add_argument("--cache", default="data/raw/alpaca_news_train_validation")
    parser.add_argument("--output-dir", default="reports/alpaca_news_research")
    args = parser.parse_args()

    base = pd.read_parquet(args.features)
    base["event_date"] = pd.to_datetime(base["event_date"]).dt.normalize()
    development = base.loc[base["event_date"].le(args.through), ["symbol", "event_date"]]
    key, secret = credentials_from_environment()
    news = collect_features(development, AlpacaNewsClient(key, secret), Path(args.cache))
    results, summary = validation_study(base, news, args.target, args.min_improvement)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    news.to_parquet(output / "train_validation_news_features.parquet", index=False)
    results.to_csv(output / "validation_results.csv", index=False)
    (output / "selection_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "ASSESSMENT.md").write_text(assessment(news, results, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
