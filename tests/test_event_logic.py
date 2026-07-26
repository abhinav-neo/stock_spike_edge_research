import numpy as np
import pandas as pd
import pytest

from src.event_study import (
    consecutive_days_above,
    extract_events_from_frame,
    first_breach,
    prepare_price_frame,
)


def test_consecutive_days_above():
    values = np.array([105, 101, 99, 110], dtype=float)
    assert consecutive_days_above(values, 100) == 2


def test_first_breach():
    values = np.array([99, 95, 89, 100], dtype=float)
    assert first_breach(values, 90) == 3


def test_prepare_price_frame_uses_adjusted_ohlc():
    frame = pd.DataFrame(
        {
            "open": [80.0, 100.0],
            "high": [90.0, 110.0],
            "low": [70.0, 95.0],
            "close": [100.0, 120.0],
            "adj_close": [200.0, 240.0],
            "volume": [1_000_000, 1_500_000],
        },
        index=pd.date_range("2020-01-01", periods=2, freq="D"),
    )

    prepared = prepare_price_frame(frame, use_adjusted_prices=True)

    assert prepared["open"].iloc[0] == 160.0
    assert prepared["high"].iloc[0] == 180.0
    assert prepared["low"].iloc[0] == 140.0
    assert prepared["close"].iloc[0] == 200.0


def test_prepare_price_frame_rejects_incomplete_adjustments():
    frame = pd.DataFrame(
        {
            "open": [80.0],
            "high": [90.0],
            "low": [70.0],
            "close": [100.0],
            "adj_close": [np.nan],
        }
    )

    with pytest.raises(ValueError, match="consistently adjusted"):
        prepare_price_frame(frame, use_adjusted_prices=True)


def test_extract_events_uses_next_day_open_and_forward_returns():
    index = pd.bdate_range("2020-01-01", periods=15)
    close = [100.0] * 10 + [150.0, 130.0, 140.0, 160.0, 170.0]
    open_ = [100.0] * 10 + [150.0, 120.0, 140.0, 160.0, 170.0]
    frame = pd.DataFrame(
        {
            "open": open_,
            "high": np.asarray(open_) + 10.0,
            "low": np.asarray(open_) - 10.0,
            "close": close,
            "adj_close": np.asarray(close) * 2.0,
            "volume": [1_000_000] * 15,
            "stock_splits": [0.0] * 15,
        },
        index=index,
    )
    cfg = {
        "minimum_history_days": 1,
        "horizons": [1, 2, 3],
        "use_adjusted_prices": True,
        "event_return_threshold": 0.40,
        "maximum_event_return": 10.0,
        "minimum_previous_close": 1.0,
        "minimum_prior_20d_avg_dollar_volume": 0.0,
        "minimum_event_day_dollar_volume": 0.0,
        "retention_levels": [1.0],
        "cooldown_days": 20,
    }

    rows = extract_events_from_frame("TEST", frame, cfg)

    assert len(rows) == 1
    row = rows[0]
    assert row["event_close"] == 300.0
    assert row["event_open"] == 300.0
    assert row["entry_open"] == 240.0
    assert row["forward_return_1d"] == pytest.approx(260.0 / 240.0 - 1)
    assert row["forward_return_2d"] == pytest.approx(280.0 / 240.0 - 1)
    assert row["forward_return_3d"] == pytest.approx(320.0 / 240.0 - 1)


def test_extract_events_respects_symbol_cooldown():
    index = pd.date_range("2020-01-01", periods=16, freq="D")
    close = [100.0] * 10 + [150.0, 100.0, 150.0, 100.0, 100.0, 100.0]
    frame = pd.DataFrame(
        {
            "open": close,
            "high": np.asarray(close) + 5.0,
            "low": np.asarray(close) - 5.0,
            "close": close,
            "adj_close": close,
            "volume": [1_000_000] * 16,
            "stock_splits": [0.0] * 16,
        },
        index=index,
    )
    cfg = {
        "minimum_history_days": 1,
        "horizons": [1],
        "use_adjusted_prices": True,
        "event_return_threshold": 0.40,
        "maximum_event_return": 10.0,
        "minimum_previous_close": 1.0,
        "minimum_prior_20d_avg_dollar_volume": 0.0,
        "minimum_event_day_dollar_volume": 0.0,
        "retention_levels": [1.0],
        "cooldown_days": 20,
    }

    rows = extract_events_from_frame("TEST", frame, cfg)

    assert len(rows) == 1
    assert rows[0]["event_date"] == index[10]

    cfg["cooldown_days"] = 0
    rows_without_cooldown = extract_events_from_frame("TEST", frame, cfg)
    assert [row["event_date"] for row in rows_without_cooldown] == [
        index[10],
        index[12],
    ]
