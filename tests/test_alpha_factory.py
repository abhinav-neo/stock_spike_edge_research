from __future__ import annotations

import pandas as pd

from src.alpha_factory import build_features, candidate_mask, candidate_specs


def sample_prices() -> pd.DataFrame:
    rows = []
    for symbol, base in [("AAA", 10.0), ("BBB", 20.0)]:
        for day in range(1, 61):
            close = base + day * 0.1
            rows.append(
                {
                    "symbol": symbol,
                    "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=day),
                    "open": close - 0.05,
                    "high": close + 0.20,
                    "low": close - 0.20,
                    "close": close,
                    "volume": 1_000_000 + day * 10_000,
                }
            )
    return pd.DataFrame(rows)


def test_build_features_preserves_rows_and_uses_prior_close() -> None:
    prices = sample_prices()
    features = build_features(prices)
    assert len(features) == len(prices)
    first_aaa = features[features["symbol"] == "AAA"].iloc[0]
    second_aaa = features[features["symbol"] == "AAA"].iloc[1]
    assert pd.isna(first_aaa["previous_close"])
    assert second_aaa["previous_close"] == first_aaa["close"]


def test_candidate_specs_cover_independent_families() -> None:
    specs = candidate_specs({"families": {}})
    families = {spec["family"] for spec in specs}
    assert families == {"gap_fade", "momentum", "mean_reversion", "breakout"}
    assert {spec["direction"] for spec in specs} == {"long", "short"}


def test_candidate_mask_returns_boolean_series() -> None:
    features = build_features(sample_prices())
    cfg = {"minimum_price": 3.0, "minimum_avg_dollar_volume": 1.0}
    spec = {
        "family": "momentum",
        "direction": "long",
        "return_threshold": 0.01,
        "rvol": 0.5,
        "ma20_distance": 0.01,
        "horizon": 5,
    }
    mask = candidate_mask(features, spec, cfg)
    assert mask.dtype == bool
    assert len(mask) == len(features)
