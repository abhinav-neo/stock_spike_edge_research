from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLUMNS = {"symbol", "asof_date"}


def validate_point_in_time_data(frame: pd.DataFrame) -> None:
    missing = KEY_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Point-in-time data missing keys: {sorted(missing)}")
    if frame.duplicated(["symbol", "asof_date"]).any():
        raise ValueError("Point-in-time data contains duplicate symbol/asof_date rows")
    value_columns = [column for column in frame.columns if column not in KEY_COLUMNS]
    if not value_columns:
        raise ValueError("Point-in-time data contains no value columns")


def asof_join_events(events: pd.DataFrame, external: pd.DataFrame, max_staleness_days: int | None = None) -> pd.DataFrame:
    validate_point_in_time_data(external)
    if not {"symbol", "event_date"}.issubset(events.columns):
        raise ValueError("Events must contain symbol and event_date")
    left = events.copy()
    right = external.copy()
    left["event_date"] = pd.to_datetime(left["event_date"]).dt.normalize()
    right["asof_date"] = pd.to_datetime(right["asof_date"]).dt.normalize()
    value_columns = [column for column in right.columns if column not in KEY_COLUMNS]
    renamed = {column: f"pit_{column}" for column in value_columns}
    frames = []
    for symbol, group in left.groupby("symbol", sort=False):
        observations = right.loc[right["symbol"] == symbol].sort_values("asof_date")
        current = group.sort_values("event_date")
        if observations.empty:
            joined = current.copy()
            joined["pit_asof_date"] = pd.NaT
            for column in renamed.values():
                joined[column] = pd.NA
        else:
            joined = pd.merge_asof(
                current,
                observations.drop(columns="symbol").rename(columns={"asof_date": "pit_asof_date", **renamed}),
                left_on="event_date",
                right_on="pit_asof_date",
                direction="backward",
                allow_exact_matches=True,
            )
        frames.append(joined)
    result = pd.concat(frames, ignore_index=True).sort_values(["event_date", "symbol"]).reset_index(drop=True)
    result["pit_staleness_days"] = (result["event_date"] - result["pit_asof_date"]).dt.days
    if (result["pit_staleness_days"].dropna() < 0).any():
        raise AssertionError("Future point-in-time observation joined to an earlier event")
    if max_staleness_days is not None:
        stale = result["pit_staleness_days"] > int(max_staleness_days)
        result.loc[stale, list(renamed.values())] = pd.NA
        result["pit_is_stale"] = stale | result["pit_asof_date"].isna()
    else:
        result["pit_is_stale"] = result["pit_asof_date"].isna()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage-safe as-of join for market cap, float, short, borrow, halt, or fundamental data")
    parser.add_argument("--events", required=True)
    parser.add_argument("--external", required=True, help="CSV or Parquet containing symbol, asof_date, and point-in-time values")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-staleness-days", type=int, default=None)
    args = parser.parse_args()
    events_path, external_path = Path(args.events), Path(args.external)
    events = pd.read_parquet(events_path) if events_path.suffix == ".parquet" else pd.read_csv(events_path)
    external = pd.read_parquet(external_path) if external_path.suffix == ".parquet" else pd.read_csv(external_path)
    joined = asof_join_events(events, external, args.max_staleness_days)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == ".parquet":
        joined.to_parquet(output, index=False)
    else:
        joined.to_csv(output, index=False)
    print(f"Wrote {len(joined):,} leakage-safe event rows to {output}")


if __name__ == "__main__":
    main()
