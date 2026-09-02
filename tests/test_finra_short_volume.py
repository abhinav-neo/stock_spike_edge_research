import pandas as pd

from src.finra_short_volume import add_finra_features, parse_finra_file


def test_parse_finra_file_filters_symbols_and_trailer():
    content = (
        b"Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        b"20240102|ABC|40|2|100|Q\n"
        b"20240102|XYZ|10|0|50|N\n"
        b"2\n"
    )
    result = parse_finra_file(content, {"ABC"})
    assert result["Symbol"].tolist() == ["ABC"]
    assert result["ShortVolume"].tolist() == [40]


def test_parse_finra_file_normalizes_symbols():
    content = (
        b"Date|Symbol|ShortVolume|ShortExemptVolume|TotalVolume|Market\n"
        b"20240102|abc|40|0|100|Q\n1\n"
    )
    assert parse_finra_file(content, {"ABC"})["Symbol"].iloc[0] == "ABC"


def test_add_finra_features_requires_exact_event_date():
    events = pd.DataFrame({"symbol": ["ABC", "ABC"], "event_date": ["2024-01-02", "2024-01-03"]})
    finra = pd.DataFrame({
        "symbol": ["ABC"], "asof_date": ["2024-01-02"],
        "finra_short_volume": [40], "finra_short_exempt_volume": [0],
        "finra_total_volume": [100], "finra_short_volume_ratio": [0.4],
    })
    result = add_finra_features(events, finra)
    assert result["finra_data_available"].tolist() == [True, False]
    assert pd.isna(result.loc[1, "finra_short_volume_ratio"])
