from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.finra_short_volume import add_finra_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Add exact-event-date FINRA features to V5 data")
    parser.add_argument("--events", required=True)
    parser.add_argument("--finra", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ratio-only", action="store_true")
    args = parser.parse_args()
    events = pd.read_parquet(args.events)
    finra = pd.read_parquet(args.finra)
    result = add_finra_features(events, finra)
    if args.ratio_only:
        result = result.drop(columns=[
            "finra_short_volume", "finra_short_exempt_volume", "finra_total_volume"
        ])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    print(f"Wrote {len(result):,} rows; FINRA coverage {result['finra_data_available'].mean():.1%}")


if __name__ == "__main__":
    main()
