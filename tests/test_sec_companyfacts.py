import io
import json
import zipfile

import pandas as pd

from src.sec_companyfacts import add_point_in_time_market_cap, parse_companyfacts_archive


def archive(payload: dict) -> bytes:
    target = io.BytesIO()
    with zipfile.ZipFile(target, "w") as output:
        output.writestr("CIK0000000123.json", json.dumps(payload))
    return target.getvalue()


def test_companyfacts_uses_filing_date_not_period_end() -> None:
    payload = {
        "facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2023-12-31", "val": 10_000_000, "form": "10-K", "filed": "2024-02-15", "accn": "a"},
            {"end": "2024-03-31", "val": 12_000_000, "form": "10-Q", "filed": "2024-05-10", "accn": "b"},
        ]}}}}}
    mapping = {"0": {"cik_str": 123, "ticker": "AAA"}}
    shares, metadata = parse_companyfacts_archive(archive(payload), mapping, {"AAA"})
    assert shares["asof_date"].tolist() == [pd.Timestamp("2024-02-15"), pd.Timestamp("2024-05-10")]
    assert metadata["symbols_with_share_facts"] == 1

    events = pd.DataFrame(
        {"symbol": ["AAA", "AAA"], "event_date": ["2024-02-01", "2024-03-01"], "event_close": [5.0, 6.0]}
    )
    features = add_point_in_time_market_cap(events, shares)
    assert pd.isna(features.loc[0, "pit_market_cap"])
    assert features.loc[1, "pit_market_cap"] == 60_000_000


def test_multi_ticker_cik_is_excluded_to_avoid_class_misattribution() -> None:
    payload = {"facts": {}}
    mapping = {
        "0": {"cik_str": 123, "ticker": "AAA"},
        "1": {"cik_str": 123, "ticker": "AAB"},
    }
    shares, metadata = parse_companyfacts_archive(archive(payload), mapping, {"AAA", "AAB"})
    assert shares.empty
    assert metadata["ambiguous_multi_ticker_ciks_excluded"] == 1
