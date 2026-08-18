import pandas as pd

from src.alpaca_intraday_research import summarize_intraday


def test_intraday_features_are_event_day_and_close_observable() -> None:
    timestamps = pd.date_range("2024-01-02 14:30:00Z", periods=60, freq="min")
    bars = pd.DataFrame({
        "symbol": "ABC", "timestamp": timestamps, "open": 10.0,
        "high": [12.0 if index == 20 else 11.0 for index in range(60)],
        "low": 9.5, "close": [10.0 + index / 100 for index in range(60)],
        "volume": 100,
    })
    events = pd.DataFrame([{"symbol": "ABC", "event_date": "2024-01-02"}])
    result = summarize_intraday(events, bars)
    assert result.loc[0, "intraday_bar_count"] == 60
    assert result.loc[0, "intraday_high_time_fraction"] == 20 / 59
    assert result.loc[0, "intraday_gap_over_5m_count"] == 0


def test_empty_intraday_response_retains_event() -> None:
    events = pd.DataFrame([{"symbol": "ABC", "event_date": "2024-01-02"}])
    result = summarize_intraday(events, pd.DataFrame())
    assert result.loc[0, "intraday_bar_count"] == 0
