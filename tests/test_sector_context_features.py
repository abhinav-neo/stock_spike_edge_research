import numpy as np
import pandas as pd

from src.sector_context_features import sector_features_for_symbol


def test_sector_inference_uses_prior_returns_not_event_spike():
    dates = pd.date_range("2023-01-02", periods=90, freq="B")
    base = pd.Series(np.linspace(100, 120, len(dates)), index=dates)
    sectors = pd.DataFrame({"UP": base, "DOWN": base.iloc[::-1].to_numpy()}, index=dates)
    stock = base.copy()
    event_date = dates[70]
    stock.loc[event_date:] = stock.loc[event_date:] * 2
    result = sector_features_for_symbol(stock, sectors, pd.Series([event_date]))
    assert result.loc[0, "inferred_sector_etf"] == "UP"
    assert result.loc[0, "prior_60d_sector_correlation"] > 0.9


def test_sector_relative_return_is_point_in_time():
    dates = pd.date_range("2023-01-02", periods=90, freq="B")
    sector = pd.Series(np.linspace(100, 110, len(dates)), index=dates)
    stock = pd.Series(np.linspace(100, 130, len(dates)), index=dates)
    sectors = pd.DataFrame({"ETF": sector}, index=dates)
    result = sector_features_for_symbol(stock, sectors, pd.Series([dates[70]]))
    expected = stock.iloc[70] / stock.iloc[50] - 1 - (sector.iloc[70] / sector.iloc[50] - 1)
    assert np.isclose(result.loc[0, "stock_minus_sector_return_20d"], expected)
