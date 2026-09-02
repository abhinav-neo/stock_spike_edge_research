import pandas as pd

from src.margin_liquidation_research import assessment


def test_assessment_rejects_profiles_that_miss_locked_gates() -> None:
    results = pd.DataFrame(
        [{
            "profile": "reg_t_50_30", "margin_liquidations": 2, "cagr": 0.10,
            "total_return": 0.20, "max_drawdown": -0.10, "sharpe_zero_rate": 0.5,
            "worst_trade": -0.2, "spy_total_return_aligned": 0.30,
        }]
    )
    report = assessment(results)
    assert "**REJECT.**" in report
    assert "10.00%" in report
