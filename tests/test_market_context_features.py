import pandas as pd
import pytest

from src.market_context_features import add_market_context, build_market_features, build_vix_features


def test_market_features_use_history_through_current_date():
    dates = pd.date_range("2023-01-02", periods=260, freq="B")
    spy = pd.DataFrame({"date": dates, "benchmark": range(100, 360)})
    features = build_market_features(spy)
    assert features.loc[5, "spy_return_5d"] == pytest.approx(105 / 100 - 1)
    assert pd.isna(features.loc[198, "spy_distance_sma_200"])
    assert pd.notna(features.loc[199, "spy_distance_sma_200"])


def test_market_join_is_exact_by_event_date():
    events = pd.DataFrame({"symbol": ["A", "B"], "event_date": ["2024-01-03", "2024-01-02"]})
    market = pd.DataFrame({"event_date": ["2024-01-02", "2024-01-03"], "spy_return_1d": [0.1, 0.2]})
    result = add_market_context(events, market)
    assert result["spy_return_1d"].tolist() == [0.2, 0.1]


def test_market_join_rejects_missing_dates():
    events = pd.DataFrame({"symbol": ["A"], "event_date": ["2024-01-03"]})
    market = pd.DataFrame({"event_date": ["2024-01-02"], "spy_return_1d": [0.1]})
    with pytest.raises(ValueError, match="Missing market context"):
        add_market_context(events, market)


def test_vix_features_use_only_current_and_prior_values():
    dates = pd.date_range("2023-01-02", periods=260, freq="B")
    vix = pd.DataFrame({"date": dates, "vix_close": range(10, 270)})
    result = build_vix_features(vix)
    assert result.loc[5, "vix_return_5d"] == pytest.approx(15 / 10 - 1)
    assert result.loc[259, "vix_percentile_252"] == pytest.approx(1.0)
