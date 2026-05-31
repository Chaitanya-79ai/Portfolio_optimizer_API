"""FastAPI entrypoint for the portfolio optimizer service.

This module wires HTTP routes to the optimizer layer and translates domain
errors into clear API status codes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.data_loader import DataLoadError, get_available_tickers, get_fund_info, load_data
from app.metrics import PortfolioDataError
from app.optimizer import (
    InfeasibleConstraintsError,
    OptimizationError,
    optimize_portfolio,
)
from app.schemas import HealthResponse, OptimizationRequest, OptimizationResponse


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the workbook during startup so data problems fail fast."""
    load_data()
    yield


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Finominal Portfolio Optimizer API",
    description="REST API for optimizing ETF portfolio allocations from historical return data.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_model=HealthResponse)
def root() -> HealthResponse:
    """Expose the same lightweight response as /health."""
    return health()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return service status and the currently supported fund tickers."""
    return HealthResponse(status="ok", available_tickers=get_available_tickers())


@app.get("/funds")
def funds() -> list[dict[str, float | str]]:
    """Return fund metadata used by clients to build valid requests."""
    fund_info = get_fund_info()
    # Display dividend yield as a percentage in the API response.
    fund_info["dividend_yield"] = (fund_info["dividend_yield"] * 100).round(4)
    return fund_info.to_dict(orient="records")


@app.post(
    "/optimize",
    response_model=OptimizationResponse,
    response_model_exclude_none=True,
)
def optimize(request: OptimizationRequest) -> dict:
    """Run a portfolio optimization request and return allocation changes."""
    tickers = [security.ticker for security in request.securities]
    weights = [security.weight for security in request.securities]

    try:
        return optimize_portfolio(
            tickers=tickers,
            current_weights=weights,
            strategy=request.strategy,
            constraints=request.constraints,
        )
    except InfeasibleConstraintsError as exc:
        # 422 means the request is valid JSON, but the constraints cannot be met.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OptimizationError, PortfolioDataError) as exc:
        # 400 covers invalid tickers, bad weights, unsupported strategies, and similar input issues.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except DataLoadError as exc:
        # Data loading failures are server-side because the workbook is an application dependency.
        raise HTTPException(status_code=500, detail=str(exc)) from exc
