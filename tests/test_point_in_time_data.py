import pandas as pd
import pytest

from src.point_in_time_data import asof_join_events, validate_point_in_time_data


def test_asof_join_never_uses_future_observation():
    events = pd.DataFrame({"symbol": ["A", "A"], "event_date": ["2024-01-05", "2024-01-15"]})
    external = pd.DataFrame({"symbol": ["A", "A"], "asof_date": ["2024-01-01", "2024-01-10"], "short_interest": [1.0, 2.0]})
    result = asof_join_events(events, external)
    assert result["pit_short_interest"].tolist() == [1.0, 2.0]
    assert (result["pit_asof_date"] <= result["event_date"]).all()


def test_stale_values_are_flagged_and_cleared():
    events = pd.DataFrame({"symbol": ["A"], "event_date": ["2024-02-01"]})
    external = pd.DataFrame({"symbol": ["A"], "asof_date": ["2024-01-01"], "borrow_bps": [500]})
    result = asof_join_events(events, external, max_staleness_days=10)
    assert result.loc[0, "pit_is_stale"]
    assert pd.isna(result.loc[0, "pit_borrow_bps"])


def test_point_in_time_data_rejects_duplicate_keys():
    external = pd.DataFrame({"symbol": ["A", "A"], "asof_date": ["2024-01-01", "2024-01-01"], "float": [1, 2]})
    with pytest.raises(ValueError, match="duplicate"):
        validate_point_in_time_data(external)
