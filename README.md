# Portfolio Optimizer API

A FastAPI backend for portfolio optimization using historical fund returns, dividend yield constraints, and factor exposure analysis.

The API accepts a portfolio of securities with current weights, applies an optimization strategy, and returns allocation changes in a clean JSON format. It supports classic portfolio construction methods such as equal weight, risk parity, minimum volatility, maximum Sharpe ratio, minimum drawdown, and factor exposure optimization.

## Features

- REST API built with FastAPI
- Historical return based portfolio optimization
- Supports long-only portfolios
- Security-level min/max weight constraints
- Portfolio-level dividend yield constraint
- Optional CAGR, volatility, and drawdown constraints
- Factor beta calculation for Value, Momentum, and Size
- Factor exposure optimizer, such as maximizing Momentum exposure
- Clear validation errors for invalid tickers, unsupported strategies, and infeasible constraints

## Tech Stack

- Python
- FastAPI
- Pandas
- NumPy
- SciPy
- OpenPyXL
- Pydantic
- Pytest
- HTTPX

## Project Structure

```text
.
├── app
│   ├── data_loader.py   # Loads and cleans Excel data
│   ├── main.py          # FastAPI routes
│   ├── metrics.py       # Portfolio metrics and factor betas
│   ├── optimizer.py     # Optimization strategies and constraints
│   └── schemas.py       # Request and response models
├── tests
├── Data.xlsx            # Local workbook
├── requirements.txt
└── README.md
```

## Data

The API reads fund and factor data from `Data.xlsx` at the project root.
The workbook is treated as a local data dependency and is ignored by git by default.

Expected sheets:

- `Fund Info`: ticker, fund name, dividend yield
- `Fund Returns`: historical daily returns per ticker
- `Factor Returns`: historical daily returns for Momentum, Value, and Size factors

Available tickers in the expected workbook:

```text
AGG, GLD, IEFA, SPY, VEA
```

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive API docs:

```text
http://127.0.0.1:8000/docs
```

## API Endpoints

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "ok",
  "available_tickers": ["AGG", "GLD", "IEFA", "SPY", "VEA"]
}
```

### List Funds

```http
GET /funds
```

Returns fund metadata including ticker, fund name, and dividend yield.

### Optimize Portfolio

```http
POST /optimize
```

Example request:

```json
{
  "strategy": "maximize_sharpe_ratio",
  "securities": [
    {"ticker": "IEFA", "weight": 20},
    {"ticker": "GLD", "weight": 20},
    {"ticker": "AGG", "weight": 20},
    {"ticker": "VEA", "weight": 20},
    {"ticker": "SPY", "weight": 20}
  ],
  "constraints": {
    "min_dividend_yield": 2.5,
    "min_weight": 5,
    "max_weight": 40
  }
}
```

Example response:

```json
{
  "optimization_strategy": "maximize_sharpe_ratio",
  "allocation_changes": [
    {
      "ticker": "IEFA",
      "security_name": "iShares Core MSCI EAFE ETF",
      "current_weight": 20.0,
      "optimized_weight": 13.76,
      "change": -6.24
    },
    {
      "ticker": "GLD",
      "security_name": "SPDR Gold Shares",
      "current_weight": 20.0,
      "optimized_weight": 5.0,
      "change": -15.0
    },
    {
      "ticker": "AGG",
      "security_name": "iShares Core US Aggregate Bond ETF",
      "current_weight": 20.0,
      "optimized_weight": 40.0,
      "change": 20.0
    },
    {
      "ticker": "VEA",
      "security_name": "Vanguard Developed Markets Index Fund;ETF",
      "current_weight": 20.0,
      "optimized_weight": 5.0,
      "change": -15.0
    },
    {
      "ticker": "SPY",
      "security_name": "State Street SPDR S&P 500 ETF Trust",
      "current_weight": 20.0,
      "optimized_weight": 36.24,
      "change": 16.24
    }
  ]
}
```

## Supported Strategies

Use one of these values in the `strategy` field:

```text
equal_weights
risk_parity
minimize_drawdown
minimize_volatility
maximize_sharpe_ratio
optimize_factor_exposure
```

## Constraint Options

The `constraints` object is optional.

Common constraints:

```json
{
  "min_weight": 5,
  "max_weight": 40,
  "min_dividend_yield": 2.5
}
```

Supported portfolio-level constraints:

```json
{
  "min_cagr": 5,
  "min_volatility": 8,
  "max_volatility": 20,
  "max_drawdown": 25
}
```

Security-specific constraints:

```json
{
  "security_constraints": [
    {"ticker": "AGG", "min_weight": 20, "max_weight": 50},
    {"ticker": "SPY", "min_weight": 10, "max_weight": 40}
  ]
}
```

Weights and percentage constraints can be provided as either percentages (`40`) or decimals (`0.40`).

## Factor Exposure

The factor exposure optimizer can maximize or minimize exposure to a target factor.
It favors cleaner exposure by penalizing unintended exposure to non-target factors.

Example request:

```json
{
  "strategy": "optimize_factor_exposure",
  "securities": [
    {"ticker": "IEFA", "weight": 20},
    {"ticker": "GLD", "weight": 20},
    {"ticker": "AGG", "weight": 20},
    {"ticker": "VEA", "weight": 20},
    {"ticker": "SPY", "weight": 20}
  ],
  "constraints": {
    "factor": "momentum",
    "direction": "maximize"
  }
}
```

Example factor beta response:

```json
{
  "factor_betas": {
    "current_portfolio": {
      "value": 0.165138,
      "momentum": 0.131574,
      "size": -0.057608
    },
    "optimized_portfolio": {
      "value": -0.030902,
      "momentum": 0.140849,
      "size": 0.041291
    }
  }
}
```

## Curl Examples

Equal weight portfolio:

```bash
curl -X POST http://127.0.0.1:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "equal_weights",
    "securities": [
      {"ticker": "IEFA", "weight": 25},
      {"ticker": "SPY", "weight": 75}
    ]
  }'
```

Minimum volatility portfolio:

```bash
curl -X POST http://127.0.0.1:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "minimize_volatility",
    "securities": [
      {"ticker": "SPY", "weight": 60},
      {"ticker": "AGG", "weight": 30},
      {"ticker": "GLD", "weight": 10}
    ]
  }'
```

Constrained Sharpe ratio optimization:

```bash
curl -X POST http://127.0.0.1:8000/optimize \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "maximize_sharpe_ratio",
    "securities": [
      {"ticker": "IEFA", "weight": 20},
      {"ticker": "GLD", "weight": 20},
      {"ticker": "AGG", "weight": 20},
      {"ticker": "VEA", "weight": 20},
      {"ticker": "SPY", "weight": 20}
    ],
    "constraints": {
      "min_dividend_yield": 2.5,
      "min_weight": 5,
      "max_weight": 40
    }
  }'
```

## How It Works

The optimizer first aligns the selected securities on their common historical return dates. It then builds a portfolio return series from the supplied weights and applies the requested optimization objective.

The optimization engine uses SciPy's SLSQP solver with long-only bounds and equality constraints so portfolio weights sum to 100%. Additional constraints are added when supplied in the request.

Sharpe ratio optimization uses arithmetic annualized returns with a 1.57% risk-free rate.

For factor betas, the API regresses portfolio returns against the provided Momentum, Value, and Size factor return series over the common date range.

## Testing

Place `Data.xlsx` at the project root before running tests. The workbook is a local data dependency and is not committed to the repository by default.

Run the full test suite:

```bash
pytest -q
```

The tests cover:

- Excel data loading and cleaning
- Expected fund and factor data availability
- Equal weight optimization
- Minimum volatility optimization
- Constrained Sharpe ratio optimization
- Infeasible constraint errors
- Factor exposure optimization
- API behavior using HTTPX with FastAPI's ASGI app

The API tests run in-process, so you do not need to start `uvicorn` before running tests.

If your environment blocks pytest cache writes, run:

```bash
PYTHONDONTWRITEBYTECODE=1 pytest -q -p no:cacheprovider
```

## Development

Run a syntax check:

```bash
python -m py_compile app/*.py
```

## Notes

- The optimizer is long-only and does not allow short selling.
- If constraints are infeasible, the API returns a validation error instead of producing invalid weights.
- `GLD` has no dividend yield in the expected workbook, so its dividend yield is treated as `0`.
