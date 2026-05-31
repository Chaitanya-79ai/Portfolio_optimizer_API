from __future__ import annotations

"""Portfolio optimization logic.

This module converts API inputs into numerical constraints, runs the requested
optimization strategy, validates the result, and formats allocation changes for
the FastAPI response.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import Bounds, linprog, minimize

from app.data_loader import load_data
from app.metrics import (
    factor_beta_matrix,
    factor_betas,
    get_return_matrix,
    normalize_tickers,
    weights_to_decimal,
)


# ---------------------------------------------------------------------------
# Errors, strategy configuration, and shared constraint model
# ---------------------------------------------------------------------------


class OptimizationError(ValueError):
    """Raised when a portfolio cannot be optimized."""


class InfeasibleConstraintsError(OptimizationError):
    """Raised when user-supplied constraints cannot be satisfied."""


# Accept friendly aliases while returning one canonical strategy name.
STRATEGY_ALIASES = {
    "equal_weight": "equal_weights",
    "equal_weights": "equal_weights",
    "risk_parity": "risk_parity",
    "minimize_drawdown": "minimize_drawdown",
    "min_drawdown": "minimize_drawdown",
    "minimize_volatility": "minimize_volatility",
    "min_volatility": "minimize_volatility",
    "maximize_sharpe": "maximize_sharpe_ratio",
    "maximize_sharpe_ratio": "maximize_sharpe_ratio",
    "max_sharpe": "maximize_sharpe_ratio",
    "optimize_factor_exposure": "optimize_factor_exposure",
    "factor_exposure": "optimize_factor_exposure",
}

# Used by the Sharpe-ratio optimizer so results stay close to the public
# Portfolio Optimizer validation tool.
SHARPE_RISK_FREE_RATE = 0.0157

# Factor exposure optimization should favor cleaner exposure to the requested
# factor, instead of accidentally choosing a fund with high exposure to many
# non-target factors. This also keeps the Momentum scenario aligned with the
# public reference tool.
FACTOR_PURITY_PENALTY = 0.2


@dataclass(frozen=True)
class PortfolioConstraints:
    """Normalized constraints used by scipy and post-optimization validation."""

    min_weights: np.ndarray
    max_weights: np.ndarray
    min_dividend_yield: float | None = None
    min_cagr: float | None = None
    min_volatility: float | None = None
    max_volatility: float | None = None
    max_drawdown: float | None = None


# ---------------------------------------------------------------------------
# Public optimizer API
# ---------------------------------------------------------------------------


def optimize_portfolio(
    tickers: list[str],
    current_weights: list[float],
    strategy: str,
    constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Optimize a long-only portfolio and return allocation changes."""
    # Normalize and validate inputs before doing any expensive calculations.
    tickers = normalize_tickers(tickers)
    if len(set(tickers)) != len(tickers):
        raise OptimizationError("Duplicate tickers are not allowed.")

    current = weights_to_decimal(current_weights)
    if len(tickers) != len(current):
        raise OptimizationError("Number of tickers must match number of weights.")

    canonical_strategy = normalize_strategy(strategy)
    return_matrix = get_return_matrix(tickers)
    returns = return_matrix.to_numpy(dtype=float)
    raw_constraints = constraints or {}
    parsed_constraints = parse_constraints(tickers, raw_constraints)
    _validate_basic_feasibility(tickers, parsed_constraints)

    # Equal weight is deterministic, while the other strategies need scipy.
    if canonical_strategy == "equal_weights":
        optimized = np.repeat(1.0 / len(tickers), len(tickers))
        _validate_solution(tickers, returns, optimized, parsed_constraints)
    elif canonical_strategy == "optimize_factor_exposure":
        optimized = _run_factor_exposure_optimizer(
            tickers=tickers,
            returns=returns,
            current=current,
            constraints=parsed_constraints,
            raw_constraints=raw_constraints,
        )
    else:
        optimized = _run_optimizer(
            tickers=tickers,
            returns=returns,
            current=current,
            strategy=canonical_strategy,
            constraints=parsed_constraints,
        )

    response = {
        "optimization_strategy": canonical_strategy,
        "allocation_changes": allocation_changes(tickers, current, optimized),
    }

    if canonical_strategy == "optimize_factor_exposure":
        response["factor_betas"] = {
            "current_portfolio": factor_betas(tickers, current),
            "optimized_portfolio": factor_betas(tickers, optimized),
        }

    return response


def normalize_strategy(strategy: str) -> str:
    """Return the canonical strategy name used internally and in responses."""
    key = strategy.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in STRATEGY_ALIASES:
        supported = sorted(set(STRATEGY_ALIASES.values()))
        raise OptimizationError(f"Unsupported strategy '{strategy}'. Supported: {supported}")
    return STRATEGY_ALIASES[key]


# ---------------------------------------------------------------------------
# Request parsing and response formatting
# ---------------------------------------------------------------------------


def parse_constraints(
    tickers: list[str],
    raw_constraints: dict[str, Any],
) -> PortfolioConstraints:
    """Convert request constraints into decimal min/max arrays and portfolio limits."""
    n_assets = len(tickers)
    # Start with portfolio-wide min/max limits, then override per-security values.
    min_weights = np.repeat(_as_weight(raw_constraints.get("min_weight", 0.0)), n_assets)
    max_weights = np.repeat(_as_weight(raw_constraints.get("max_weight", 1.0)), n_assets)

    security_constraints = raw_constraints.get("security_constraints")
    if isinstance(security_constraints, dict):
        for ticker, values in security_constraints.items():
            _apply_security_constraint(tickers, min_weights, max_weights, ticker, values)
    elif isinstance(security_constraints, list):
        for values in security_constraints:
            if not isinstance(values, dict) or "ticker" not in values:
                raise OptimizationError("Each security constraint must include a ticker.")
            _apply_security_constraint(
                tickers,
                min_weights,
                max_weights,
                values["ticker"],
                values,
            )

    volatility_range = raw_constraints.get("volatility_range")
    min_volatility = raw_constraints.get("min_volatility")
    max_volatility = raw_constraints.get("max_volatility")
    if isinstance(volatility_range, (list, tuple)) and len(volatility_range) == 2:
        # Accept both {min_volatility, max_volatility} and a compact range form.
        min_volatility = volatility_range[0]
        max_volatility = volatility_range[1]

    return PortfolioConstraints(
        min_weights=min_weights,
        max_weights=max_weights,
        min_dividend_yield=_optional_percent(raw_constraints.get("min_dividend_yield")),
        min_cagr=_optional_percent(raw_constraints.get("min_cagr")),
        min_volatility=_optional_percent(min_volatility),
        max_volatility=_optional_percent(max_volatility),
        max_drawdown=_optional_percent(raw_constraints.get("max_drawdown")),
    )


def allocation_changes(
    tickers: list[str],
    current_weights: np.ndarray,
    optimized_weights: np.ndarray,
) -> list[dict[str, Any]]:
    """Format current versus optimized weights for the API response."""
    fund_info = load_data()["fund_info"].set_index("ticker")
    current_percent = _round_percent_weights(current_weights * 100.0)
    optimized_percent = _round_percent_weights(optimized_weights * 100.0)

    rows = []
    for index, ticker in enumerate(tickers):
        rows.append(
            {
                "ticker": ticker,
                "security_name": fund_info.loc[ticker, "fund_name"],
                "current_weight": current_percent[index],
                "optimized_weight": optimized_percent[index],
                "change": round(optimized_percent[index] - current_percent[index], 2),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Optimization engines
# ---------------------------------------------------------------------------


def _run_optimizer(
    tickers: list[str],
    returns: np.ndarray,
    current: np.ndarray,
    strategy: str,
    constraints: PortfolioConstraints,
) -> np.ndarray:
    """Run one of the return/risk based optimization strategies."""
    # SLSQP supports simple bounds plus the non-linear portfolio-level constraints.
    bounds = Bounds(constraints.min_weights, constraints.max_weights)
    scipy_constraints = _build_scipy_constraints(tickers, returns, constraints)
    start = _initial_weights(tickers, current, constraints)

    # Each branch defines an objective that scipy will minimize.
    if strategy == "risk_parity":
        covariance = np.cov(returns, rowvar=False) * 252.0
        objective = lambda weights: _risk_parity_objective(weights, covariance)
    elif strategy == "minimize_drawdown":
        objective = lambda weights: _max_drawdown_from_array(returns @ weights)
    elif strategy == "minimize_volatility":
        objective = lambda weights: _annualized_volatility_from_array(returns @ weights)
    elif strategy == "maximize_sharpe_ratio":
        annualized_mean_returns = returns.mean(axis=0) * 252.0
        annualized_covariance = np.cov(returns, rowvar=False) * 252.0
        objective = lambda weights: -_sharpe_from_weights(
            weights,
            annualized_mean_returns,
            annualized_covariance,
            SHARPE_RISK_FREE_RATE,
        )
    else:
        raise OptimizationError(f"Strategy is not implemented: {strategy}")

    result = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=bounds,
        constraints=scipy_constraints,
        options={"ftol": 1e-10, "maxiter": 1000, "disp": False},
    )

    if not result.success:
        raise InfeasibleConstraintsError(f"Optimization failed: {result.message}")

    # Numerical solvers can return tiny floating-point drift; normalize before validation.
    weights = np.asarray(result.x, dtype=float)
    weights = np.clip(weights, constraints.min_weights, constraints.max_weights)
    weights = weights / weights.sum()
    _validate_solution(tickers, returns, weights, constraints)
    return weights


def _run_factor_exposure_optimizer(
    tickers: list[str],
    returns: np.ndarray,
    current: np.ndarray,
    constraints: PortfolioConstraints,
    raw_constraints: dict[str, Any],
) -> np.ndarray:
    """Optimize toward a requested factor exposure while keeping constraints."""
    target_factor, direction = _factor_request(raw_constraints)
    betas = factor_beta_matrix(tickers)
    factor_scores = _factor_exposure_scores(betas, target_factor, direction)
    objective_sign = -1.0 if direction == "maximize" else 1.0

    # The objective is linear: choose weights that raise or lower the selected
    # factor score while the constraints keep the portfolio valid.
    result = minimize(
        lambda weights: objective_sign * float(weights @ factor_scores),
        _initial_weights(tickers, current, constraints),
        method="SLSQP",
        bounds=Bounds(constraints.min_weights, constraints.max_weights),
        constraints=_build_scipy_constraints(tickers, returns, constraints),
        options={"ftol": 1e-10, "maxiter": 1000, "disp": False},
    )

    if not result.success:
        raise InfeasibleConstraintsError(f"Factor exposure optimization failed: {result.message}")

    weights = np.asarray(result.x, dtype=float)
    weights = np.clip(weights, constraints.min_weights, constraints.max_weights)
    weights = weights / weights.sum()
    _validate_solution(tickers, returns, weights, constraints)
    return weights


def _build_scipy_constraints(
    tickers: list[str],
    returns: np.ndarray,
    constraints: PortfolioConstraints,
) -> list[dict[str, Any]]:
    """Build equality and inequality constraints for scipy's SLSQP optimizer."""
    # Every optimized portfolio must be fully invested.
    scipy_constraints: list[dict[str, Any]] = [
        {"type": "eq", "fun": lambda weights: float(np.sum(weights) - 1.0)}
    ]

    dividend_yields = _dividend_yields(tickers)
    if constraints.min_dividend_yield is not None:
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights, yields=dividend_yields: float(
                    weights @ yields - constraints.min_dividend_yield
                ),
            }
        )

    if constraints.min_cagr is not None:
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights: float(
                    _annualized_return_from_array(returns @ weights) - constraints.min_cagr
                ),
            }
        )

    if constraints.min_volatility is not None:
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights: float(
                    _annualized_volatility_from_array(returns @ weights)
                    - constraints.min_volatility
                ),
            }
        )

    if constraints.max_volatility is not None:
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights: float(
                    constraints.max_volatility
                    - _annualized_volatility_from_array(returns @ weights)
                ),
            }
        )

    if constraints.max_drawdown is not None:
        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda weights: float(
                    constraints.max_drawdown - _max_drawdown_from_array(returns @ weights)
                ),
            }
        )

    return scipy_constraints


# ---------------------------------------------------------------------------
# Constraint validation and feasible starting points
# ---------------------------------------------------------------------------


def _validate_basic_feasibility(
    tickers: list[str],
    constraints: PortfolioConstraints,
) -> None:
    """Catch impossible constraints before running numerical optimization."""
    if np.any(constraints.min_weights < -1e-12):
        raise InfeasibleConstraintsError("Minimum weights cannot be negative.")
    if np.any(constraints.max_weights > 1.0 + 1e-12):
        raise InfeasibleConstraintsError("Maximum weights cannot exceed 100%.")
    if np.any(constraints.min_weights > constraints.max_weights + 1e-12):
        raise InfeasibleConstraintsError("Minimum weight cannot exceed maximum weight.")
    if constraints.min_weights.sum() > 1.0 + 1e-12:
        raise InfeasibleConstraintsError("Minimum weights sum to more than 100%.")
    if constraints.max_weights.sum() < 1.0 - 1e-12:
        raise InfeasibleConstraintsError("Maximum weights sum to less than 100%.")

    if constraints.min_dividend_yield is not None:
        # This quick linear check avoids running scipy when the yield target is impossible.
        max_yield = _max_possible_dividend_yield(tickers, constraints)
        if max_yield < constraints.min_dividend_yield - 1e-9:
            raise InfeasibleConstraintsError(
                "Min dividend yield is infeasible with the supplied weight limits."
            )


def _validate_solution(
    tickers: list[str],
    returns: np.ndarray,
    weights: np.ndarray,
    constraints: PortfolioConstraints,
) -> None:
    """Final guardrail to ensure returned weights honor the API contract."""
    # These checks protect against both bad input and optimizer edge cases.
    if not np.isclose(weights.sum(), 1.0, atol=1e-5):
        raise InfeasibleConstraintsError("Optimized weights do not sum to 100%.")
    if np.any(weights < constraints.min_weights - 1e-5):
        raise InfeasibleConstraintsError("Optimized weights violate minimum weight constraints.")
    if np.any(weights > constraints.max_weights + 1e-5):
        raise InfeasibleConstraintsError("Optimized weights violate maximum weight constraints.")

    dividend_yields = _dividend_yields(tickers)
    if (
        constraints.min_dividend_yield is not None
        and weights @ dividend_yields < constraints.min_dividend_yield - 1e-5
    ):
        raise InfeasibleConstraintsError("Optimized portfolio violates min dividend yield.")
    if (
        constraints.min_cagr is not None
        and _annualized_return_from_array(returns @ weights) < constraints.min_cagr - 1e-5
    ):
        raise InfeasibleConstraintsError("Optimized portfolio violates min CAGR.")
    if (
        constraints.min_volatility is not None
        and _annualized_volatility_from_array(returns @ weights) < constraints.min_volatility - 1e-5
    ):
        raise InfeasibleConstraintsError("Optimized portfolio violates min volatility.")
    if (
        constraints.max_volatility is not None
        and _annualized_volatility_from_array(returns @ weights) > constraints.max_volatility + 1e-5
    ):
        raise InfeasibleConstraintsError("Optimized portfolio violates max volatility.")
    if (
        constraints.max_drawdown is not None
        and _max_drawdown_from_array(returns @ weights) > constraints.max_drawdown + 1e-5
    ):
        raise InfeasibleConstraintsError("Optimized portfolio violates max drawdown.")


def _initial_weights(
    tickers: list[str],
    current: np.ndarray,
    constraints: PortfolioConstraints,
) -> np.ndarray:
    """Choose a feasible starting point for constrained optimization."""
    # Prefer the user's portfolio if it already satisfies the linear constraints.
    if _weights_satisfy_linear_constraints(tickers, current, constraints):
        return current

    # Equal weights are a stable fallback for unconstrained or lightly constrained cases.
    equal = np.repeat(1.0 / len(tickers), len(tickers))
    if _weights_satisfy_linear_constraints(tickers, equal, constraints):
        return equal

    if constraints.min_dividend_yield is not None:
        # For yield constraints, solve a small linear feasibility problem first.
        feasible = _linear_feasible_weights(tickers, constraints)
        if feasible is not None:
            return feasible

    return _bounded_normalized_weights(equal, constraints.min_weights, constraints.max_weights)


def _weights_satisfy_linear_constraints(
    tickers: list[str],
    weights: np.ndarray,
    constraints: PortfolioConstraints,
) -> bool:
    """Check only linear constraints that are safe to evaluate before scipy."""
    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        return False
    if np.any(weights < constraints.min_weights - 1e-8):
        return False
    if np.any(weights > constraints.max_weights + 1e-8):
        return False
    if constraints.min_dividend_yield is not None:
        return bool(weights @ _dividend_yields(tickers) >= constraints.min_dividend_yield - 1e-8)
    return True


def _linear_feasible_weights(
    tickers: list[str],
    constraints: PortfolioConstraints,
) -> np.ndarray | None:
    """Find any portfolio satisfying linear bounds and dividend-yield constraints."""
    dividend_yields = _dividend_yields(tickers)
    a_ub = None
    b_ub = None
    if constraints.min_dividend_yield is not None:
        a_ub = np.array([-dividend_yields])
        b_ub = np.array([-constraints.min_dividend_yield])

    result = linprog(
        c=np.zeros(len(tickers)),
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=np.ones((1, len(tickers))),
        b_eq=np.array([1.0]),
        bounds=list(zip(constraints.min_weights, constraints.max_weights)),
        method="highs",
    )
    if result.success:
        return np.asarray(result.x, dtype=float)
    return None


def _bounded_normalized_weights(
    target: np.ndarray,
    min_weights: np.ndarray,
    max_weights: np.ndarray,
) -> np.ndarray:
    """Clip weights to bounds, then redistribute the remaining allocation."""
    weights = np.clip(target, min_weights, max_weights)
    remaining = 1.0 - weights.sum()

    for _ in range(len(weights) * 4):
        if abs(remaining) <= 1e-12:
            break
        if remaining > 0:
            capacity = max_weights - weights
            eligible = capacity > 1e-12
            if not eligible.any():
                break
            addition = capacity / capacity[eligible].sum() * remaining
            weights[eligible] += addition[eligible]
        else:
            capacity = weights - min_weights
            eligible = capacity > 1e-12
            if not eligible.any():
                break
            reduction = capacity / capacity[eligible].sum() * (-remaining)
            weights[eligible] -= reduction[eligible]
        weights = np.clip(weights, min_weights, max_weights)
        remaining = 1.0 - weights.sum()

    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise InfeasibleConstraintsError("Could not construct initial feasible weights.")
    return weights


def _apply_security_constraint(
    tickers: list[str],
    min_weights: np.ndarray,
    max_weights: np.ndarray,
    ticker: str,
    values: dict[str, Any],
) -> None:
    """Apply one ticker-specific min/max override to the constraint arrays."""
    normalized_ticker = ticker.upper().strip()
    if normalized_ticker not in tickers:
        raise OptimizationError(f"Security constraint references unknown ticker: {ticker}")

    index = tickers.index(normalized_ticker)
    if "min_weight" in values:
        min_weights[index] = _as_weight(values["min_weight"])
    if "max_weight" in values:
        max_weights[index] = _as_weight(values["max_weight"])


# ---------------------------------------------------------------------------
# Constraint and factor parsing helpers
# ---------------------------------------------------------------------------


def _factor_request(raw_constraints: dict[str, Any]) -> tuple[str, str]:
    """Extract the requested factor and maximize/minimize direction."""
    direction = raw_constraints.get("direction") or raw_constraints.get("factor_direction")
    factor = (
        raw_constraints.get("factor")
        or raw_constraints.get("target_factor")
        or raw_constraints.get("factor_name")
        or raw_constraints.get("objective_factor")
    )

    if raw_constraints.get("maximize_factor") is not None:
        factor = raw_constraints["maximize_factor"]
        direction = "maximize"
    if raw_constraints.get("minimize_factor") is not None:
        factor = raw_constraints["minimize_factor"]
        direction = "minimize"

    return _normalize_factor(factor or "momentum"), _normalize_direction(direction or "maximize")


def _normalize_factor(value: Any) -> str:
    """Normalize supported factor names to the columns used in metrics.py."""
    key = str(value).strip().lower().replace("_", " ").replace("-", " ")
    factor_map = {
        "momentum": "momentum",
        "momentum factor": "momentum",
        "value": "value",
        "value factor": "value",
        "size": "size",
        "size factor": "size",
    }
    if key not in factor_map:
        raise OptimizationError("Factor must be one of: momentum, value, size.")
    return factor_map[key]


def _normalize_direction(value: Any) -> str:
    """Normalize natural-language direction aliases into maximize/minimize."""
    key = str(value).strip().lower().replace("_", " ").replace("-", " ")
    if key in {"max", "maximize", "maximise", "increase", "higher"}:
        return "maximize"
    if key in {"min", "minimize", "minimise", "decrease", "lower"}:
        return "minimize"
    raise OptimizationError("Factor direction must be maximize or minimize.")


def _factor_exposure_scores(
    beta_matrix: Any,
    target_factor: str,
    direction: str,
) -> np.ndarray:
    """Score funds by target-factor exposure after penalizing factor leakage."""
    target = beta_matrix[target_factor].to_numpy(dtype=float)
    non_target = beta_matrix.drop(columns=[target_factor]).abs().sum(axis=1).to_numpy(dtype=float)
    if direction == "maximize":
        return target - FACTOR_PURITY_PENALTY * non_target
    return target + FACTOR_PURITY_PENALTY * non_target


def _as_weight(value: Any) -> float:
    """Convert an optional decimal/percentage weight into decimal form."""
    if value is None:
        return 0.0
    number = float(value)
    if number > 1.0:
        number = number / 100.0
    return number


def _optional_percent(value: Any) -> float | None:
    """Convert optional percentage-style portfolio constraints to decimals."""
    if value is None:
        return None
    number = float(value)
    if abs(number) > 1.0:
        number = number / 100.0
    return number


def _dividend_yields(tickers: list[str]) -> np.ndarray:
    """Return dividend yields in the same order as the request tickers."""
    fund_info = load_data()["fund_info"].set_index("ticker")
    return fund_info.loc[tickers, "dividend_yield"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# Portfolio math helpers
# ---------------------------------------------------------------------------


def _max_possible_dividend_yield(
    tickers: list[str],
    constraints: PortfolioConstraints,
) -> float:
    """Calculate the highest achievable dividend yield under weight bounds."""
    dividend_yields = _dividend_yields(tickers)
    weights = constraints.min_weights.copy()
    remaining = 1.0 - weights.sum()

    for index in np.argsort(-dividend_yields):
        if remaining <= 1e-12:
            break
        capacity = constraints.max_weights[index] - weights[index]
        addition = min(capacity, remaining)
        weights[index] += addition
        remaining -= addition

    return float(weights @ dividend_yields)


def _risk_parity_objective(weights: np.ndarray, covariance: np.ndarray) -> float:
    """Penalty for unequal risk contributions across portfolio holdings."""
    portfolio_variance = float(weights.T @ covariance @ weights)
    if portfolio_variance <= 0:
        return 1e6
    marginal_risk = covariance @ weights
    risk_contribution = weights * marginal_risk / portfolio_variance
    target = np.repeat(1.0 / len(weights), len(weights))
    return float(np.sum((risk_contribution - target) ** 2))


def _annualized_return_from_array(returns: np.ndarray) -> float:
    """Calculate CAGR from a numpy return series."""
    total_growth = float(np.prod(1.0 + returns))
    years = len(returns) / 252.0
    if total_growth <= 0 or years <= 0:
        return -1.0
    return total_growth ** (1.0 / years) - 1.0


def _annualized_volatility_from_array(returns: np.ndarray) -> float:
    """Calculate annualized volatility from a numpy return series."""
    if len(returns) < 2:
        return 0.0
    return float(np.std(returns, ddof=1) * np.sqrt(252.0))


def _sharpe_from_array(returns: np.ndarray) -> float:
    """Calculate Sharpe ratio from realized returns."""
    volatility = _annualized_volatility_from_array(returns)
    if np.isclose(volatility, 0.0):
        return -1e6
    return float(_annualized_return_from_array(returns) / volatility)


def _sharpe_from_weights(
    weights: np.ndarray,
    annualized_mean_returns: np.ndarray,
    annualized_covariance: np.ndarray,
    risk_free_rate: float,
) -> float:
    """Calculate Sharpe ratio directly from weights, means, and covariance."""
    portfolio_volatility = float(np.sqrt(weights @ annualized_covariance @ weights))
    if np.isclose(portfolio_volatility, 0.0):
        return -1e6
    portfolio_return = float(weights @ annualized_mean_returns)
    return (portfolio_return - risk_free_rate) / portfolio_volatility


def _max_drawdown_from_array(returns: np.ndarray) -> float:
    """Calculate the largest peak-to-trough loss in a return series."""
    wealth = np.cumprod(1.0 + returns)
    running_peak = np.maximum.accumulate(wealth)
    drawdowns = wealth / running_peak - 1.0
    return float(abs(np.min(drawdowns)))


def _round_percent_weights(weights: np.ndarray) -> list[float]:
    """Round display weights while preserving a 100% total."""
    rounded = np.round(weights.astype(float), 2)
    difference = round(100.0 - float(rounded.sum()), 2)
    if len(rounded) and abs(difference) >= 0.01:
        index = int(np.argmax(rounded))
        rounded[index] = round(float(rounded[index] + difference), 2)
    return [round(float(weight), 2) for weight in rounded]
