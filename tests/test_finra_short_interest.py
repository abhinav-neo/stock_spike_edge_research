import pandas as pd

from src.finra_short_interest import attach_short_interest, normalize_short_interest


def test_normalization_applies_conservative_publication_lag() -> None:
    result = normalize_short_interest(
        [{
            "symbolCode": "ABC", "settlementDate": "2024-01-15",
            "currentShortPositionQuantity": 1000, "previousShortPositionQuantity": 800,
            "averageDailyVolumeQuantity": 200, "daysToCoverQuantity": 5,
            "stockSplitFlag": None,
        }]
    )
    assert result.loc[0, "asof_date"] == pd.Timestamp("2024-01-29")
    assert result.loc[0, "short_interest_to_adv"] == 5


def test_event_before_publication_cannot_see_settlement() -> None:
    external = normalize_short_interest(
        [{
            "symbolCode": "ABC", "settlementDate": "2024-01-15",
            "currentShortPositionQuantity": 1000, "previousShortPositionQuantity": 800,
            "averageDailyVolumeQuantity": 200, "daysToCoverQuantity": 5,
        }]
    )
    events = pd.DataFrame(
        [{"symbol": "ABC", "event_date": "2024-01-28"}, {"symbol": "ABC", "event_date": "2024-01-29"}]
    )
    result = attach_short_interest(events, external)
    assert not result.loc[0, "short_interest_data_available"]
    assert result.loc[1, "short_interest_data_available"]
