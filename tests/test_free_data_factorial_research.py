import pandas as pd

from src.free_data_factorial_research import all_combinations, build_variant, feature_group


def test_all_combinations_has_power_set():
    combinations = all_combinations(["a", "b", "c"])
    assert len(combinations) == 8
    assert () in combinations and ("a", "b", "c") in combinations


def test_feature_group_selects_only_named_prefix():
    frame = pd.DataFrame({"symbol": ["ABC"], "event_date": ["2024-01-01"], "spy_return_1d": [0.1], "vix_close": [20]})
    result = feature_group(frame, "spy")
    assert result.columns.tolist() == ["symbol", "event_date", "spy_return_1d"]


def test_build_variant_preserves_rows_and_keys():
    base = pd.DataFrame({"symbol": ["ABC"], "event_date": ["2024-01-01"], "x": [1]})
    group = pd.DataFrame({"symbol": ["ABC"], "event_date": ["2024-01-01"], "spy_return_1d": [0.1]})
    result = build_variant(base, {"spy": group}, ("spy",))
    assert len(result) == 1
    assert result.loc[0, "spy_return_1d"] == 0.1
