from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from src.atomic_io import atomic_write_parquet
from src.point_in_time_data import asof_join_events


SHARES_TAG = "EntityCommonStockSharesOutstanding"
PERIODIC_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "40-F", "40-F/A"}


def ticker_mapping(payload: dict) -> tuple[dict[str, str], set[str]]:
    """Map unambiguous CIKs to tickers and report multi-ticker CIKs."""
    rows = payload.values() if isinstance(payload, dict) else payload
    by_cik: dict[str, set[str]] = {}
    for row in rows:
        cik = str(row.get("cik_str", row.get("cik", ""))).zfill(10)
        ticker = str(row.get("ticker", "")).strip().upper()
        if cik.strip("0") and ticker:
            by_cik.setdefault(cik, set()).add(ticker)
    ambiguous = {cik for cik, tickers in by_cik.items() if len(tickers) != 1}
    return {cik: next(iter(tickers)) for cik, tickers in by_cik.items() if cik not in ambiguous}, ambiguous


def extract_share_facts(payload: dict, symbol: str) -> pd.DataFrame:
    facts = payload.get("facts", {}).get("dei", {}).get(SHARES_TAG, {}).get("units", {}).get("shares", [])
    rows = []
    for fact in facts:
        if fact.get("form") not in PERIODIC_FORMS:
            continue
        filed = pd.to_datetime(fact.get("filed"), errors="coerce")
        period_end = pd.to_datetime(fact.get("end"), errors="coerce")
        value = pd.to_numeric(fact.get("val"), errors="coerce")
        if pd.isna(filed) or pd.isna(period_end) or pd.isna(value) or float(value) <= 0:
            continue
        rows.append(
            {
                "symbol": symbol,
                "asof_date": filed.normalize(),
                "shares_outstanding": float(value),
                "shares_period_end": period_end.normalize(),
                "shares_form": fact.get("form"),
                "shares_accession": fact.get("accn"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["symbol", "asof_date", "shares_outstanding", "shares_period_end", "shares_form", "shares_accession"]
        )
    frame = pd.DataFrame(rows).sort_values(["asof_date", "shares_period_end", "shares_accession"])
    return frame.drop_duplicates(["symbol", "asof_date"], keep="last").reset_index(drop=True)


def parse_companyfacts_archive(content: bytes, mapping_payload: dict, wanted_symbols: set[str]) -> tuple[pd.DataFrame, dict]:
    cik_map, ambiguous = ticker_mapping(mapping_payload)
    wanted = {symbol.upper() for symbol in wanted_symbols}
    frames = []
    parsed = 0
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for name in archive.namelist():
            stem = Path(name).stem.upper()
            if not stem.startswith("CIK"):
                continue
            cik = stem.removeprefix("CIK").zfill(10)
            symbol = cik_map.get(cik)
            if symbol not in wanted:
                continue
            payload = json.loads(archive.read(name))
            frame = extract_share_facts(payload, symbol)
            if not frame.empty:
                frames.append(frame)
            parsed += 1
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not result.empty:
        result = result.sort_values(["symbol", "asof_date"]).reset_index(drop=True)
    metadata = {
        "symbols_requested": len(wanted),
        "companies_parsed": parsed,
        "symbols_with_share_facts": int(result["symbol"].nunique()) if len(result) else 0,
        "ambiguous_multi_ticker_ciks_excluded": len(ambiguous),
        "availability_rule": "SEC filing date; never fiscal period end",
        "limitation": "Current SEC ticker mappings do not reconstruct historical symbol changes or delisted mappings.",
    }
    return result, metadata


def add_point_in_time_market_cap(events: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    joined = asof_join_events(events, shares, max_staleness_days=550)
    price_column = "event_close" if "event_close" in joined else "close"
    if price_column not in joined:
        raise ValueError("Events require event_close or close to derive market capitalization")
    joined["pit_market_cap"] = pd.to_numeric(joined[price_column], errors="coerce") * pd.to_numeric(
        joined["pit_shares_outstanding"], errors="coerce"
    )
    joined.loc[joined["pit_is_stale"], "pit_market_cap"] = pd.NA
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(description="Build filing-date-safe SEC shares and market-cap features")
    parser.add_argument("--companyfacts-zip", required=True)
    parser.add_argument("--ticker-map", required=True)
    parser.add_argument("--events", default="data/processed/events_features_v5.parquet")
    parser.add_argument("--shares-output", default="data/processed/sec_shares_point_in_time.parquet")
    parser.add_argument("--features-output", default="data/processed/events_features_sec_market_cap.parquet")
    parser.add_argument("--metadata", default="reports/sec_companyfacts/metadata.json")
    args = parser.parse_args()

    events = pd.read_parquet(args.events)
    mapping = json.loads(Path(args.ticker_map).read_text(encoding="utf-8"))
    shares, metadata = parse_companyfacts_archive(
        Path(args.companyfacts_zip).read_bytes(), mapping, set(events["symbol"].astype(str))
    )
    features = add_point_in_time_market_cap(events, shares)
    atomic_write_parquet(shares, Path(args.shares_output))
    atomic_write_parquet(features, Path(args.features_output))
    metadata["event_coverage"] = float(features["pit_market_cap"].notna().mean())
    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
