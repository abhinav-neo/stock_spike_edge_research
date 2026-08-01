from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


BASE_URL = "https://cdn.finra.org/equity/regsho/daily"
USER_AGENT = "stock-spike-edge-research/1.0 (non-commercial research)"


def parse_finra_file(content: bytes, wanted_symbols: set[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(io.BytesIO(content), sep="|", dtype={"Symbol": str})
    required = {"Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Unexpected FINRA columns: {list(frame.columns)}")
    frame = frame.loc[pd.to_numeric(frame["Date"], errors="coerce").notna()].copy()
    frame["Symbol"] = frame["Symbol"].str.upper()
    if wanted_symbols is not None:
        frame = frame.loc[frame["Symbol"].isin(wanted_symbols)]
    for column in ["ShortVolume", "ShortExemptVolume", "TotalVolume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _download(url: str, retries: int = 3) -> bytes | None:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in {403, 404}:
                return None
            if attempt + 1 == retries:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt + 1 == retries:
                raise
        time.sleep(0.5 * (attempt + 1))
    return None


def fetch_date(date: pd.Timestamp, symbols: set[str]) -> tuple[pd.DataFrame, str]:
    stamp = pd.Timestamp(date).strftime("%Y%m%d")
    consolidated = _download(f"{BASE_URL}/CNMSshvol{stamp}.txt")
    source = "CNMS"
    pieces: list[pd.DataFrame] = []
    if consolidated is not None:
        pieces.append(parse_finra_file(consolidated, symbols))
    else:
        source = "FNSQ+FNYX"
        for prefix in ("FNSQ", "FNYX"):
            content = _download(f"{BASE_URL}/{prefix}shvol{stamp}.txt")
            if content is not None:
                pieces.append(parse_finra_file(content, symbols))
    if not pieces:
        return pd.DataFrame(), "missing"
    frame = pd.concat(pieces, ignore_index=True)
    if frame.empty:
        return frame, source
    grouped = frame.groupby("Symbol", as_index=False)[["ShortVolume", "ShortExemptVolume", "TotalVolume"]].sum(min_count=1)
    grouped["asof_date"] = pd.Timestamp(date).normalize()
    grouped["source"] = source
    return grouped, source


def collect_for_events(events: pd.DataFrame, workers: int = 6) -> tuple[pd.DataFrame, dict]:
    if not {"symbol", "event_date"}.issubset(events.columns):
        raise ValueError("Events must contain symbol and event_date")
    work = events[["symbol", "event_date"]].copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    work["event_date"] = pd.to_datetime(work["event_date"]).dt.normalize()
    symbols_by_date = {date: set(group["symbol"]) for date, group in work.groupby("event_date")}
    frames: list[pd.DataFrame] = []
    source_counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_date, date, symbols): date for date, symbols in symbols_by_date.items()}
        for future in as_completed(futures):
            frame, source = future.result()
            source_counts[source] = source_counts.get(source, 0) + 1
            if not frame.empty:
                frames.append(frame)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if raw.empty:
        result = pd.DataFrame(columns=["symbol", "asof_date", "finra_short_volume", "finra_short_exempt_volume", "finra_total_volume", "finra_short_volume_ratio", "finra_source"])
    else:
        result = raw.rename(columns={
            "Symbol": "symbol", "ShortVolume": "finra_short_volume",
            "ShortExemptVolume": "finra_short_exempt_volume", "TotalVolume": "finra_total_volume",
            "source": "finra_source",
        })
        denominator = result["finra_total_volume"].replace(0, np.nan)
        result["finra_short_volume_ratio"] = result["finra_short_volume"] / denominator
        result = result.sort_values(["asof_date", "symbol"]).reset_index(drop=True)
    matched = work.merge(result[["symbol", "asof_date"]], left_on=["symbol", "event_date"], right_on=["symbol", "asof_date"], how="left")["asof_date"].notna()
    metadata = {
        "license": "FINRA data free for non-commercial use; subject to FINRA Terms of Use",
        "semantic_warning": "Short-sale transaction volume is not short interest, borrow cost, or borrow availability.",
        "requested_event_rows": int(len(work)),
        "requested_dates": int(len(symbols_by_date)),
        "matched_event_rows": int(matched.sum()),
        "coverage": float(matched.mean()),
        "source_date_counts": source_counts,
    }
    return result, metadata


def add_finra_features(events: pd.DataFrame, finra: pd.DataFrame) -> pd.DataFrame:
    """Exact-date join: event-close FINRA data may inform a next-session entry."""
    work = events.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    work["event_date"] = pd.to_datetime(work["event_date"]).dt.normalize()
    external = finra.copy()
    external["symbol"] = external["symbol"].astype(str).str.upper()
    external["asof_date"] = pd.to_datetime(external["asof_date"]).dt.normalize()
    if external.duplicated(["symbol", "asof_date"]).any():
        raise ValueError("Duplicate FINRA symbol/asof_date rows")
    numeric = [
        "finra_short_volume", "finra_short_exempt_volume",
        "finra_total_volume", "finra_short_volume_ratio",
    ]
    available = [column for column in numeric if column in external.columns]
    joined = work.merge(
        external[["symbol", "asof_date", *available]],
        left_on=["symbol", "event_date"], right_on=["symbol", "asof_date"],
        how="left", validate="one_to_one",
    ).drop(columns="asof_date")
    joined["finra_data_available"] = joined["finra_short_volume_ratio"].notna()
    return joined


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect free FINRA daily short-sale volume for event dates")
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    path = Path(args.events)
    events = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    data, metadata = collect_for_events(events, workers=args.workers)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(output, index=False)
    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
