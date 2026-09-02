from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.atomic_io import atomic_write_csv


LOCATE_COLUMNS = [
    "observation_id", "decision_timestamp", "provider", "locate_requested", "locate_confirmed",
    "quoted_borrow_rate_annual", "available_quantity", "source_reference",
    "locate_basis",
]


def validate_locate_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(LOCATE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Locate evidence missing columns: {sorted(missing)}")
    result = frame[LOCATE_COLUMNS].copy()
    if result["observation_id"].isna().any() or result["observation_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Locate evidence requires observation_id")
    if result["observation_id"].duplicated().any():
        raise ValueError("Locate evidence contains duplicate observation_id decisions")
    result["decision_timestamp"] = pd.to_datetime(result["decision_timestamp"], utc=True, errors="coerce")
    if result["decision_timestamp"].isna().any():
        raise ValueError("Locate evidence contains invalid decision timestamps")
    if result["provider"].isna().any() or result["provider"].astype(str).str.strip().eq("").any():
        raise ValueError("Locate evidence requires a named provider")
    for column in ("locate_requested", "locate_confirmed"):
        result[column] = result[column].fillna(False).astype(bool)
    allowed_basis = {"explicit_locate", "broker_etb"}
    if not set(result["locate_basis"].dropna().astype(str)).issubset(allowed_basis):
        raise ValueError(f"Locate basis must be one of {sorted(allowed_basis)}")
    explicit = result["locate_basis"].eq("explicit_locate")
    if (result["locate_confirmed"] & explicit & ~result["locate_requested"]).any():
        raise ValueError("A locate cannot be confirmed when it was not requested")
    for column in ("quoted_borrow_rate_annual", "available_quantity"):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    if (result["locate_confirmed"] & explicit & result["available_quantity"].fillna(0).le(0)).any():
        raise ValueError("Confirmed locates require positive available quantity")
    if result["quoted_borrow_rate_annual"].dropna().lt(0).any():
        raise ValueError("Quoted borrow rates must be non-negative")
    return result.sort_values(["decision_timestamp", "observation_id"]).reset_index(drop=True)


def append_locate_evidence(existing: pd.DataFrame, additions: pd.DataFrame) -> pd.DataFrame:
    frames = [frame for frame in (existing, additions) if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=LOCATE_COLUMNS)
    return validate_locate_evidence(pd.concat(frames, ignore_index=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and append actual broker locate decisions")
    parser.add_argument("--input", required=True, help="Provider export using the required locate-evidence schema")
    parser.add_argument("--output", default="reports/forward_observation/locate_evidence.csv")
    args = parser.parse_args()

    source = pd.read_csv(args.input)
    output = Path(args.output)
    existing = pd.read_csv(output) if output.exists() and output.stat().st_size else pd.DataFrame()
    combined = append_locate_evidence(existing, source)
    atomic_write_csv(combined, output)
    print(f"Validated actual locate decisions: {len(combined)}. No orders were submitted.")


if __name__ == "__main__":
    main()
