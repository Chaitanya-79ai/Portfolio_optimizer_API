"""Portfolio analytics used by the optimizer and tests.

This module keeps reusable calculations separate from optimization strategy
selection: return alignment, weight conversion, risk/return metrics, dividend
yield, and factor beta regression.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.data_loader import load_data


# ---------------------------------------------------------------------------
# Constants and errors
# ---------------------------------------------------------------------------


TRADING_DAYS_PER_YEAR = 252
FACTOR_COLUMNS = ["momentum", "value", "size"]
FACTOR_OUTPUT_ORDER = ["value", "momentum", "size"]
# Aligns optimizer inputs with the available live-tool comparison window.
PORTFOLIO_HISTORY_START_DATE = pd.Timestamp("2007-07-26")


class PortfolioDataError(ValueError):
    """Raised when portfolio inputs cannot be evaluated with available data."""


# ---------------------------------------------------------------------------
# Input normalization and return matrices
# ---------------------------------------------------------------------------


def normalize_tickers(tickers: list[str]) -> list[str]:
    """Normalize tickers using the same convention as the data loader."""
    return [ticker.upper().strip() for ticker in tickers]


def weights_to_decimal(weights: list[float] | np.ndarray) -> np.ndarray:
    """Accept weights as either percentages summing to 100 or decimals summing to 1."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1:
        raise PortfolioDataError("Weights must be a one-dimensional list.")

    total = values.sum()
    if np.isclose(total, 100.0, atol=1e-6):
        values = values / 100.0
    elif not np.isclose(total, 1.0, atol=1e-6):
        # The API accepts either 20/80 style inputs or 0.2/0.8 style inputs.
        raise PortfolioDataError("Weights must sum to 100 or 1.")

    if np.any(values < -1e-9):
        raise PortfolioDataError("Weights cannot be negative.")

    return values


def get_return_matrix(tickers: list[str]) -> pd.DataFrame:
    """Return aligned daily returns for the requested tickers."""
    tickers = normalize_tickers(tickers)
    data = load_data()
    fund_returns = data["fund_returns"]

    available = set(data["fund_info"]["ticker"])
    invalid = sorted(set(tickers) - available)
    if invalid:
        # Keep the message explicit so API clients know exactly which symbol failed.
        raise PortfolioDataError(f"Unsupported tickers: {invalid}")

    filtered = fund_returns[fund_returns["ticker"].isin(tickers)]
    matrix = filtered.pivot_table(
        index="date",
        columns="ticker",
        values="total_return",
        aggfunc="first",
    )
    # Preserve request order so weight vectors and response rows stay aligned.
    matrix = matrix.reindex(columns=tickers)
    # Use only dates where every requested ticker has data.
    matrix = matrix[matrix.index >= PORTFOLIO_HISTORY_START_DATE].dropna()

    if matrix.empty:
        raise PortfolioDataError("No common return history found for selected tickers.")

    return matrix


def get_factor_return_matrix() -> pd.DataFrame:
    """Return factor returns with normalized factor names."""
    factor_returns = load_data()["factor_returns"]
    name_map = {
        "Momentum Factor": "momentum",
        "Value Factor": "value",
        "Size Factor": "size",
    }
    matrix = factor_returns.pivot_table(
        index="date",
        columns="index_ticker",
        values="total_return",
        aggfunc="first",
    )
    matrix = matrix.rename(columns=name_map)
    return matrix[FACTOR_COLUMNS].dropna()


# ---------------------------------------------------------------------------
# Portfolio-level metrics
# ---------------------------------------------------------------------------


def portfolio_returns(return_matrix: pd.DataFrame, weights: list[float] | np.ndarray) -> pd.Series:
    """Combine asset returns into a single weighted portfolio return series."""
    decimal_weights = weights_to_decimal(weights)
    if len(decimal_weights) != len(return_matrix.columns):
        raise PortfolioDataError("Number of weights must match number of return columns.")

    returns = return_matrix.dot(decimal_weights)
    returns.name = "portfolio_return"
    return returns


def annualized_return(returns: pd.Series) -> float:
    """Calculate CAGR from a pandas return series."""
    returns = returns.dropna()
    if returns.empty:
        raise PortfolioDataError("Cannot calculate annualized return with no returns.")

    total_growth = float((1.0 + returns).prod())
    years = len(returns) / TRADING_DAYS_PER_YEAR
    if years <= 0:
        raise PortfolioDataError("Return history is too short.")

    return total_growth ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series) -> float:
    """Calculate annualized volatility from a pandas return series."""
    returns = returns.dropna()
    if len(returns) < 2:
        raise PortfolioDataError("At least two returns are required for volatility.")
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio from realized returns."""
    volatility = annualized_volatility(returns)
    if np.isclose(volatility, 0.0):
        return 0.0
    return float((annualized_return(returns) - risk_free_rate) / volatility)


def max_drawdown(returns: pd.Series) -> float:
    """Calculate the largest historical peak-to-trough decline."""
    returns = returns.dropna()
    if returns.empty:
        raise PortfolioDataError("Cannot calculate drawdown with no returns.")

    wealth = (1.0 + returns).cumprod()
    running_peak = wealth.cummax()
    drawdowns = wealth / running_peak - 1.0
    return float(abs(drawdowns.min()))


def portfolio_dividend_yield(tickers: list[str], weights: list[float] | np.ndarray) -> float:
    """Calculate weighted dividend yield for a portfolio."""
    tickers = normalize_tickers(tickers)
    decimal_weights = weights_to_decimal(weights)
    if len(tickers) != len(decimal_weights):
        raise PortfolioDataError("Number of tickers must match number of weights.")

    fund_info = load_data()["fund_info"].set_index("ticker")
    yields = fund_info.loc[tickers, "dividend_yield"].to_numpy(dtype=float)
    return float(np.dot(decimal_weights, yields))


# ---------------------------------------------------------------------------
# Factor beta calculations
# ---------------------------------------------------------------------------


def factor_betas(tickers: list[str], weights: list[float] | np.ndarray) -> dict[str, float]:
    """Estimate portfolio factor betas with an intercept-based linear regression."""
    return_matrix = get_return_matrix(tickers)
    portfolio = portfolio_returns(return_matrix, weights)
    factor_matrix = get_factor_return_matrix()

    aligned = pd.concat([portfolio, factor_matrix], axis=1, join="inner").dropna()
    if len(aligned) < 5:
        # Regression needs enough overlapping observations to produce meaningful betas.
        raise PortfolioDataError("Not enough overlapping history for factor regression.")

    y = aligned["portfolio_return"].to_numpy(dtype=float)
    x = aligned[FACTOR_COLUMNS].to_numpy(dtype=float)
    # Include alpha so betas represent factor sensitivity, not the intercept.
    x_with_alpha = np.column_stack([np.ones(len(x)), x])

    coefficients, *_ = np.linalg.lstsq(x_with_alpha, y, rcond=None)
    return _format_factor_betas(coefficients[1:])


def factor_beta_matrix(tickers: list[str]) -> pd.DataFrame:
    """Estimate factor betas for each individual ticker in a portfolio."""
    tickers = normalize_tickers(tickers)
    return_matrix = get_return_matrix(tickers)
    factor_matrix = get_factor_return_matrix()

    aligned = pd.concat([return_matrix, factor_matrix], axis=1, join="inner").dropna()
    if len(aligned) < 5:
        raise PortfolioDataError("Not enough overlapping history for factor regression.")

    x = aligned[FACTOR_COLUMNS].to_numpy(dtype=float)
    x_with_alpha = np.column_stack([np.ones(len(x)), x])

    rows = []
    for ticker in tickers:
        # Regress each asset against the same factor history.
        y = aligned[ticker].to_numpy(dtype=float)
        coefficients, *_ = np.linalg.lstsq(x_with_alpha, y, rcond=None)
        rows.append(coefficients[1:])

    return pd.DataFrame(rows, index=tickers, columns=FACTOR_COLUMNS)


def weighted_factor_betas(
    tickers: list[str],
    weights: list[float] | np.ndarray,
) -> dict[str, float]:
    """Calculate weighted factor betas from precomputed individual security betas."""
    decimal_weights = weights_to_decimal(weights)
    beta_matrix = factor_beta_matrix(tickers)
    weighted = beta_matrix.to_numpy(dtype=float).T @ decimal_weights
    return _format_factor_betas(weighted)


def _format_factor_betas(beta_values: np.ndarray) -> dict[str, float]:
    """Round and order beta values for stable API responses."""
    beta_by_factor = {
        factor: round(float(beta_values[index]), 6)
        for index, factor in enumerate(FACTOR_COLUMNS)
    }
    return {factor: beta_by_factor[factor] for factor in FACTOR_OUTPUT_ORDER}
