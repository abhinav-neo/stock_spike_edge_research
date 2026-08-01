from __future__ import annotations

import argparse
import io
import json
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


BASE_URL = "https://www.sec.gov/files/data/fails-deliver-data"
LEGACY_BASE_URL = "https://www.sec.gov/files/data/frequently-requested-foia-document-fails-deliver-data"
NODE_BASE_URL = "https://www.sec.gov/files/node/add/data_distribution"
USER_AGENT = "stock-spike-edge-research abhinav.neo@gmail.com"


def conservative_publication_date(settlement_date: pd.Timestamp) -> pd.Timestamp:
    """First safe date after the SEC's stated half-month publication window."""
    date = pd.Timestamp(settlement_date).normalize()
    if date.day <= 15:
        return (date + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)).normalize()
    return (date + pd.offsets.MonthEnd(0) + pd.Timedelta(days=16)).normalize()


def parse_sec_zip(content: bytes, wanted_symbols: set[str] | None = None) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if len(names) != 1:
            raise ValueError(f"Expected one SEC data file, found {names}")
        raw = archive.read(names[0])
    lines = raw.decode("latin-1").splitlines()
    if not lines or not lines[0].upper().startswith("SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)"):
        raise ValueError("Unexpected SEC FTD header")
    # Some issuer descriptions contain a literal pipe. Required fields are fixed at
    # the front and price is last, so parse them positionally without discarding rows.
    records = []
    for line in lines[1:]:
        parts = line.split("|")
        if len(parts) >= 6:
            records.append((parts[0], parts[2], parts[3], parts[-1]))
    frame = pd.DataFrame(records, columns=["SETTLEMENT DATE", "SYMBOL", "QUANTITY (FAILS)", "PRICE"])
    frame["symbol"] = frame["SYMBOL"].str.strip().str.upper()
    if wanted_symbols is not None:
        frame = frame.loc[frame["symbol"].isin(wanted_symbols)]
    frame["settlement_date"] = pd.to_datetime(frame["SETTLEMENT DATE"], format="%Y%m%d", errors="coerce")
    frame["ftd_quantity"] = pd.to_numeric(frame["QUANTITY (FAILS)"], errors="coerce")
    frame["ftd_price"] = pd.to_numeric(frame["PRICE"], errors="coerce")
    frame = frame.loc[frame["settlement_date"].notna() & frame["ftd_quantity"].notna()].copy()
    frame["available_date"] = frame["settlement_date"].map(conservative_publication_date)
    frame["ftd_dollar_value"] = frame["ftd_quantity"] * frame["ftd_price"]
    return frame[["symbol", "settlement_date", "available_date", "ftd_quantity", "ftd_price", "ftd_dollar_value"]]


def _download(url: str, cache_path: Path | None = None, retries: int = 8) -> bytes | None:
    if cache_path is not None and cache_path.exists():
        return cache_path.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    for attempt in range(retries):
        time.sleep(0.15)
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                content = response.read()
                if cache_path is not None:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_bytes(content)
                return content
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                return None
            if exc.code == 429 and attempt + 1 < retries:
                retry_after = exc.headers.get("Retry-After")
                time.sleep(float(retry_after) if retry_after and retry_after.isdigit() else min(30, 2 ** (attempt + 1)))
                continue
            if attempt + 1 == retries:
                raise
        except (TimeoutError, urllib.error.URLError):
            if attempt + 1 == retries:
                raise
        time.sleep(attempt + 1)
    return None


def _download_archive(filename: str, cache_path: Path | None = None) -> bytes | None:
    candidates = [
        f"{LEGACY_BASE_URL}/{filename}", f"{BASE_URL}/{filename}", f"{NODE_BASE_URL}/{filename}",
    ]
    if filename == "cnsfails201910a.zip":
        candidates.append(f"{BASE_URL}/cnsfails201910a_0.zip")
    for url in candidates:
        content = _download(url, cache_path)
        if content is not None:
            return content
    return None


def archive_specs(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[str, str]]:
    months = pd.period_range(pd.Timestamp(start).to_period("M"), pd.Timestamp(end).to_period("M"), freq="M")
    return [(f"{period.year}{period.month:02d}{half}", f"cnsfails{period.year}{period.month:02d}{half}.zip") for period in months for half in ("a", "b")]


def collect_history(symbols: set[str], start: pd.Timestamp, end: pd.Timestamp, workers: int = 1, cache_dir: Path | None = None) -> tuple[pd.DataFrame, dict]:
    specs = archive_specs(start, end)
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_download_archive, filename, cache_dir / filename if cache_dir else None): key
            for key, filename in specs
        }
        for future in as_completed(futures):
            key = futures[future]
            content = future.result()
            if content is None:
                missing.append(key)
            else:
                frame = parse_sec_zip(content, symbols)
                if not frame.empty:
                    frames.append(frame)
    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not result.empty:
        result = result.sort_values(["symbol", "available_date", "settlement_date"]).reset_index(drop=True)
    metadata = {
        "source": "SEC Fails-to-Deliver Data",
        "semantic_warning": "Fails can arise from long or short sales and are not borrow availability or proof of short selling.",
        "publication_lag": "First-half records usable from first day of next month; second-half records from day 16 of next month.",
        "archives_requested": len(specs), "archives_missing": sorted(missing),
        "symbols_requested": len(symbols), "rows_retained": int(len(result)),
    }
    return result, metadata


def add_ftd_features(events: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    work["symbol"] = work["symbol"].astype(str).str.upper()
    work["event_date"] = pd.to_datetime(work["event_date"]).dt.normalize().astype("datetime64[ns]")
    external = history.copy()
    external["available_date"] = pd.to_datetime(external["available_date"]).dt.normalize().astype("datetime64[ns]")
    external["settlement_date"] = pd.to_datetime(external["settlement_date"]).dt.normalize().astype("datetime64[ns]")
    rows: list[pd.DataFrame] = []
    for symbol, group in work.groupby("symbol", sort=False):
        observations = external.loc[external["symbol"] == symbol].sort_values(["available_date", "settlement_date"])
        # Multiple settlement dates become public together; use the latest one and summarize the released batch.
        if not observations.empty:
            releases = observations.groupby("available_date", as_index=False).agg(
                ftd_latest_settlement_date=("settlement_date", "max"),
                ftd_quantity=("ftd_quantity", "last"),
                ftd_dollar_value=("ftd_dollar_value", "last"),
                ftd_release_observations=("ftd_quantity", "size"),
                ftd_release_mean_quantity=("ftd_quantity", "mean"),
                ftd_release_max_quantity=("ftd_quantity", "max"),
            )
            joined = pd.merge_asof(
                group.sort_values("event_date"), releases.sort_values("available_date"),
                left_on="event_date", right_on="available_date", direction="backward", allow_exact_matches=True,
            )
        else:
            joined = group.copy()
            for column in ["available_date", "ftd_latest_settlement_date", "ftd_quantity", "ftd_dollar_value", "ftd_release_observations", "ftd_release_mean_quantity", "ftd_release_max_quantity"]:
                joined[column] = pd.NaT if "date" in column else np.nan
        rows.append(joined)
    result = pd.concat(rows, ignore_index=True).sort_values(["event_date", "symbol"]).reset_index(drop=True)
    result["ftd_data_available"] = result["available_date"].notna()
    result["ftd_publication_staleness_days"] = (result["event_date"] - result["available_date"]).dt.days
    result["ftd_settlement_staleness_days"] = (result["event_date"] - result["ftd_latest_settlement_date"]).dt.days
    for column in ["ftd_quantity", "ftd_dollar_value", "ftd_release_mean_quantity", "ftd_release_max_quantity"]:
        result[f"{column}_log1p"] = np.log1p(result[column].clip(lower=0))
    if (result["ftd_publication_staleness_days"].dropna() < 0).any():
        raise AssertionError("Future SEC publication joined to an earlier event")
    return result.drop(columns=["available_date", "ftd_latest_settlement_date"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect SEC FTD history and build leakage-safe event features")
    parser.add_argument("--events", required=True)
    parser.add_argument("--history-output", required=True)
    parser.add_argument("--features-output", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--cache-dir", default="data/raw/sec_ftd_archives")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    events = pd.read_parquet(args.events)
    dates = pd.to_datetime(events["event_date"])
    history, metadata = collect_history(
        set(events["symbol"].astype(str).str.upper()), dates.min() - pd.offsets.MonthBegin(2), dates.max(),
        args.workers, Path(args.cache_dir),
    )
    features = add_ftd_features(events, history)
    for path, frame in [(Path(args.history_output), history), (Path(args.features_output), features)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    metadata["event_rows"] = int(len(features))
    metadata["event_coverage"] = float(features["ftd_data_available"].mean())
    metadata_path = Path(args.metadata)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
