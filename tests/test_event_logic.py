\
import numpy as np

from src.event_study import consecutive_days_above, first_breach


def test_consecutive_days_above():
    values = np.array([105, 101, 99, 110], dtype=float)
    assert consecutive_days_above(values, 100) == 2


def test_first_breach():
    values = np.array([99, 95, 89, 100], dtype=float)
    assert first_breach(values, 90) == 3
