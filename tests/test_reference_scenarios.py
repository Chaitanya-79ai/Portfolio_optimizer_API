from app.metrics import portfolio_dividend_yield
from app.optimizer import InfeasibleConstraintsError, optimize_portfolio


REQUIRED_RESPONSE_KEYS = {
    "ticker",
    "security_name",
    "current_weight",
    "optimized_weight",
    "change",
}


def _assert_allocation_response(result, expected_tickers):
    rows = result["allocation_changes"]
    weights = [row["optimized_weight"] for row in rows]

    assert [row["ticker"] for row in rows] == expected_tickers
    assert round(sum(weights), 2) == 100.0
    assert all(weight >= 0 for weight in weights)

    for row in rows:
        assert REQUIRED_RESPONSE_KEYS <= set(row)
        assert isinstance(row["security_name"], str)
        assert row["security_name"]


def test_case_1_equal_weight_sanity_check():
    tickers = ["IEFA", "SPY"]
    result = optimize_portfolio(tickers, [25, 75], "equal_weights")

    _assert_allocation_response(result, tickers)
    assert [row["optimized_weight"] for row in result["allocation_changes"]] == [50.0, 50.0]


def test_case_2_risk_parity():
    tickers = ["VEA", "AGG"]
    result = optimize_portfolio(tickers, [25, 75], "risk_parity")

    _assert_allocation_response(result, tickers)


def test_case_3_minimize_volatility():
    tickers = ["SPY", "AGG", "GLD"]
    result = optimize_portfolio(tickers, [60, 30, 10], "minimize_volatility")

    _assert_allocation_response(result, tickers)


def test_case_4_maximize_sharpe_ratio():
    tickers = ["IEFA", "GLD", "AGG", "VEA", "SPY"]
    result = optimize_portfolio(tickers, [20, 20, 20, 20, 20], "maximize_sharpe_ratio")

    _assert_allocation_response(result, tickers)


def test_case_5_constraints_are_respected():
    tickers = ["IEFA", "GLD", "AGG", "VEA", "SPY"]
    result = optimize_portfolio(
        tickers,
        [20, 20, 20, 20, 20],
        "maximize_sharpe_ratio",
        {"min_dividend_yield": 2.5, "min_weight": 5, "max_weight": 40},
    )
    weights = [row["optimized_weight"] for row in result["allocation_changes"]]

    _assert_allocation_response(result, tickers)
    assert min(weights) >= 5.0
    assert max(weights) <= 40.0
    assert portfolio_dividend_yield(tickers, weights) * 100 >= 2.5 - 1e-3


def test_case_6_factor_exposure_increases_momentum_and_returns_betas():
    tickers = ["IEFA", "GLD", "AGG", "VEA", "SPY"]
    result = optimize_portfolio(
        tickers,
        [20, 20, 20, 20, 20],
        "optimize_factor_exposure",
        {"maximize_factor": "momentum"},
    )

    _assert_allocation_response(result, tickers)
    weights = {row["ticker"]: row["optimized_weight"] for row in result["allocation_changes"]}
    assert weights["GLD"] == 100.0
    assert "factor_betas" in result
    current = result["factor_betas"]["current_portfolio"]
    optimized = result["factor_betas"]["optimized_portfolio"]
    assert set(current) == {"value", "momentum", "size"}
    assert set(optimized) == {"value", "momentum", "size"}
    assert optimized["momentum"] > current["momentum"]


def test_infeasible_constraints_return_clear_error():
    try:
        optimize_portfolio(
            ["IEFA", "GLD", "AGG", "VEA", "SPY"],
            [20, 20, 20, 20, 20],
            "maximize_sharpe_ratio",
            {"min_dividend_yield": 10, "min_weight": 5, "max_weight": 40},
        )
    except InfeasibleConstraintsError as exc:
        assert "infeasible" in str(exc).lower()
    else:
        raise AssertionError("Expected infeasible constraints to raise a clear error.")
