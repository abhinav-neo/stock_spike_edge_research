import pandas as pd

from src.analyze_edges import candidate_masks, load_parameter_grid, parameter_stability


def sample_events():
    return pd.DataFrame(
        {
            "event_return": [0.50, 0.50],
            "close_location": [0.20, 0.80],
            "relative_dollar_volume": [3.0, 3.0],
            "event_close": [6.0, 6.0],
        }
    )


def test_custom_parameter_grid_controls_candidate_count():
    cfg = {
        "parameter_grid": {
            "continuation_return_bands": [[0.40, 0.60]],
            "continuation_close_locations": [0.75],
            "failed_spike_close_locations": [0.25, 0.40],
            "relative_volumes": [2],
            "minimum_prices": [5],
        }
    }
    candidates = candidate_masks(sample_events(), cfg)
    assert len(candidates) == 3
    assert sum(c["side"] == "long" for c in candidates) == 1
    assert sum(c["side"] == "short" for c in candidates) == 2


def test_candidate_metadata_is_preserved():
    cfg = {
        "parameter_grid": {
            "continuation_return_bands": [[0.40, 0.60]],
            "continuation_close_locations": [0.75],
            "failed_spike_close_locations": [0.25],
            "relative_volumes": [2],
            "minimum_prices": [5],
        }
    }
    failed = [c for c in candidate_masks(sample_events(), cfg) if c["side"] == "short"][0]
    assert failed["close_location"] == 0.25
    assert failed["relative_volume"] == 2.0
    assert failed["minimum_price"] == 5.0
    assert failed["mask"].tolist() == [True, False]


def test_parameter_grid_rejects_empty_values():
    cfg = {"parameter_grid": {"relative_volumes": []}}
    try:
        load_parameter_grid(cfg)
    except ValueError as exc:
        assert "relative_volumes" in str(exc)
    else:
        raise AssertionError("Expected an empty grid dimension to fail")


def test_parameter_stability_aggregates_parameter_regions():
    edges = pd.DataFrame(
        {
            "rule": ["a", "b"],
            "side": ["short", "short"],
            "close_location": [0.4, 0.4],
            "relative_volume": [2.0, 5.0],
            "minimum_price": [5.0, 5.0],
            "horizon": [20, 20],
            "sample_size_pass": [True, False],
            "robust_score": [0.2, 0.1],
            "test_mean_return": [0.1, 0.05],
            "test_win_rate": [0.6, 0.55],
            "test_t_stat": [2.0, 1.0],
        }
    )
    stability = parameter_stability(edges)
    close_rows = stability[stability["parameter"] == "close_location"]
    assert len(close_rows) == 1
    assert close_rows.iloc[0]["rules"] == 2
    assert close_rows.iloc[0]["passing_rules"] == 1
