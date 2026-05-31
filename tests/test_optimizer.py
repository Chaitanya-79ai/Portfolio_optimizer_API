import pytest

from app.metrics import portfolio_dividend_yield
from app.optimizer import InfeasibleConstraintsError, optimize_portfolio


def _optimized_weights(result):
    return [row["optimized_weight"] for row in result["allocation_changes"]]


def test_equal_weights_strategy_splits_allocation_equally():
    result = optimize_portfolio(
        tickers=["IEFA", "SPY"],
        current_weights=[25, 75],
        strategy="equal_weights",
    )

    assert result["optimization_strategy"] == "equal_weights"
    assert _optimized_weights(result) == [50.0, 50.0]


def test_minimize_volatility_returns_valid_long_only_weights():
    result = optimize_portfolio(
        tickers=["SPY", "AGG", "GLD"],
        current_weights=[60, 30, 10],
        strategy="minimize_volatility",
    )
    weights = _optimized_weights(result)

    assert round(sum(weights), 2) == 100.0
    assert all(weight >= 0 for weight in weights)


def test_maximize_sharpe_matches_reference_case():
    result = optimize_portfolio(
        tickers=["IEFA", "GLD", "AGG", "VEA", "SPY"],
        current_weights=[20, 20, 20, 20, 20],
        strategy="maximize_sharpe_ratio",
    )

    assert _optimized_weights(result) == [0.0, 30.61, 0.0, 0.0, 69.39]


def test_constrained_sharpe_respects_weight_and_yield_limits():
    tickers = ["IEFA", "GLD", "AGG", "VEA", "SPY"]
    result = optimize_portfolio(
        tickers=tickers,
        current_weights=[20, 20, 20, 20, 20],
        strategy="maximize_sharpe_ratio",
        constraints={
            "min_dividend_yield": 2.5,
            "min_weight": 5,
            "max_weight": 40,
        },
    )
    weights = _optimized_weights(result)

    assert round(sum(weights), 2) == 100.0
    assert min(weights) >= 5.0
    assert max(weights) <= 40.0
    assert portfolio_dividend_yield(tickers, weights) * 100 >= 2.5 - 1e-3


def test_factor_exposure_optimizer_increases_momentum_beta():
    result = optimize_portfolio(
        tickers=["IEFA", "GLD", "AGG", "VEA", "SPY"],
        current_weights=[20, 20, 20, 20, 20],
        strategy="optimize_factor_exposure",
        constraints={"maximize_factor": "momentum"},
    )

    assert "factor_betas" in result
    current = result["factor_betas"]["current_portfolio"]
    optimized = result["factor_betas"]["optimized_portfolio"]
    assert optimized["momentum"] > current["momentum"]


def test_infeasible_constraints_raise_clear_error():
    with pytest.raises(InfeasibleConstraintsError, match="Min dividend yield is infeasible"):
        optimize_portfolio(
            tickers=["IEFA", "GLD", "AGG", "VEA", "SPY"],
            current_weights=[20, 20, 20, 20, 20],
            strategy="maximize_sharpe_ratio",
            constraints={
                "min_dividend_yield": 10,
                "min_weight": 5,
                "max_weight": 40,
            },
        )
