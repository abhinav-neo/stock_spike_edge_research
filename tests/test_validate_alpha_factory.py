from __future__ import annotations

import numpy as np
import pandas as pd

from src.validate_alpha_factory import benjamini_hochberg, executable_forward_return


def test_executable_forward_return_enters_next_open() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "date": pd.date_range("2024-01-01", periods=3),
            "open": [10.0, 20.0, 30.0],
            "close": [11.0, 22.0, 33.0],
        }
    )
    result = executable_forward_return(frame, 1, "long")
    assert np.isclose(result.iloc[0], 22.0 / 20.0 - 1.0)
    assert np.isclose(result.iloc[1], 33.0 / 30.0 - 1.0)
    assert pd.isna(result.iloc[2])


def test_short_return_is_negative_of_long_return() -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "AAA"],
            "date": pd.date_range("2024-01-01", periods=3),
            "open": [10.0, 20.0, 30.0],
            "close": [11.0, 18.0, 27.0],
        }
    )
    long_result = executable_forward_return(frame, 1, "long")
    short_result = executable_forward_return(frame, 1, "short")
    assert np.allclose(short_result.dropna(), -long_result.dropna())


def test_benjamini_hochberg_controls_multiple_tests() -> None:
    q_values, passed = benjamini_hochberg(pd.Series([0.001, 0.02, 0.50]), 0.05)
    assert passed.tolist() == [True, True, False]
    assert q_values.iloc[0] <= q_values.iloc[1] <= q_values.iloc[2]
