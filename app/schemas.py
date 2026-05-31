"""Pydantic models for API request validation and response serialization."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class SecurityInput(BaseModel):
    """One holding supplied by the client."""

    ticker: str = Field(..., min_length=1)
    weight: float = Field(..., ge=0)

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        """Normalize ticker symbols before the optimizer sees them."""
        return value.upper().strip()


class OptimizationRequest(BaseModel):
    """Request body accepted by POST /optimize."""

    model_config = ConfigDict(
        populate_by_name=True,
        # This example appears in FastAPI's generated Swagger documentation.
        json_schema_extra={
            "example": {
                "strategy": "maximize_sharpe_ratio",
                "securities": [
                    {"ticker": "IEFA", "weight": 20},
                    {"ticker": "GLD", "weight": 20},
                    {"ticker": "AGG", "weight": 20},
                    {"ticker": "VEA", "weight": 20},
                    {"ticker": "SPY", "weight": 20},
                ],
                "constraints": {
                    "min_dividend_yield": 2.5,
                    "min_weight": 5,
                    "max_weight": 40,
                },
            }
        },
    )

    strategy: str = Field(..., min_length=1)
    securities: list[SecurityInput] = Field(..., min_length=1)
    constraints: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def accept_optimization_strategy_alias(cls, values: Any) -> Any:
        # Accept assignment-style input while keeping internal code on `strategy`.
        if isinstance(values, dict) and "strategy" not in values:
            if "optimization_strategy" in values:
                values = dict(values)
                values["strategy"] = values["optimization_strategy"]
        return values

    @field_validator("securities")
    @classmethod
    def reject_duplicate_tickers(cls, securities: list[SecurityInput]) -> list[SecurityInput]:
        """Reject duplicate holdings because weights would otherwise be ambiguous."""
        tickers = [security.ticker for security in securities]
        if len(tickers) != len(set(tickers)):
            raise ValueError("Duplicate tickers are not allowed.")
        return securities


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class AllocationChange(BaseModel):
    """Current and optimized allocation for one security."""

    ticker: str
    security_name: str
    current_weight: float
    optimized_weight: float
    change: float


class FactorBetas(BaseModel):
    """Factor exposure values returned for the optional factor comparison."""

    value: float
    momentum: float
    size: float


class FactorBetaComparison(BaseModel):
    """Before/after factor beta comparison for factor exposure requests."""

    current_portfolio: FactorBetas
    optimized_portfolio: FactorBetas


class OptimizationResponse(BaseModel):
    """Response returned by POST /optimize."""

    optimization_strategy: str
    allocation_changes: list[AllocationChange]
    factor_betas: FactorBetaComparison | None = None


class HealthResponse(BaseModel):
    """Response returned by health-style endpoints."""

    status: str
    available_tickers: list[str]
