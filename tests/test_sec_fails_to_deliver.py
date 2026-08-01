import io
import zipfile

import pandas as pd

from src.sec_fails_to_deliver import add_ftd_features, conservative_publication_date, parse_sec_zip


def _zip(payload: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("data.txt", payload)
    return output.getvalue()


def test_conservative_publication_lag():
    assert conservative_publication_date(pd.Timestamp("2024-01-10")) == pd.Timestamp("2024-02-01")
    assert conservative_publication_date(pd.Timestamp("2024-01-25")) == pd.Timestamp("2024-02-16")


def test_parse_sec_zip_filters_and_computes_value():
    content = _zip(b"SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20240110|1|ABC|100|ABC INC|2.5\n20240110|2|XYZ|20|XYZ INC|5\n")
    result = parse_sec_zip(content, {"ABC"})
    assert result["symbol"].tolist() == ["ABC"]
    assert result["ftd_dollar_value"].tolist() == [250.0]


def test_parse_sec_zip_tolerates_pipe_in_description():
    content = _zip(b"SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20240110|1|ABC|100|ABC|OLD NAME|2.5\n")
    result = parse_sec_zip(content, {"ABC"})
    assert result["ftd_price"].tolist() == [2.5]


def test_asof_join_uses_publication_not_settlement_date():
    events = pd.DataFrame({"symbol": ["ABC", "ABC"], "event_date": ["2024-01-20", "2024-02-02"]})
    history = pd.DataFrame({
        "symbol": ["ABC"], "settlement_date": ["2024-01-10"], "available_date": ["2024-02-01"],
        "ftd_quantity": [100], "ftd_price": [2.5], "ftd_dollar_value": [250.0],
    })
    result = add_ftd_features(events, history)
    assert result["ftd_data_available"].tolist() == [False, True]
    assert pd.isna(result.loc[0, "ftd_quantity"])
    assert result.loc[1, "ftd_publication_staleness_days"] == 1
