import asyncio

import httpx

from app.main import app


def _request(method: str, path: str, **kwargs) -> httpx.Response:
    async def send_request() -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await getattr(client, method)(path, **kwargs)

    return asyncio.run(send_request())


def test_health_endpoint_returns_available_tickers():
    response = _request("get", "/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["available_tickers"] == ["AGG", "GLD", "IEFA", "SPY", "VEA"]


def test_optimize_endpoint_returns_allocation_changes():
    response = _request(
        "post",
        "/optimize",
        json={
            "strategy": "equal_weights",
            "securities": [
                {"ticker": "IEFA", "weight": 25},
                {"ticker": "SPY", "weight": 75},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["optimization_strategy"] == "equal_weights"
    assert body["allocation_changes"][0]["ticker"] == "IEFA"
    assert body["allocation_changes"][0]["optimized_weight"] == 50.0


def test_optimize_endpoint_maps_bad_ticker_to_http_error():
    response = _request(
        "post",
        "/optimize",
        json={
            "strategy": "equal_weights",
            "securities": [{"ticker": "BAD", "weight": 100}],
        },
    )

    assert response.status_code == 400
    assert "Unsupported tickers" in response.json()["detail"]


def test_factor_exposure_endpoint_returns_factor_betas():
    response = _request(
        "post",
        "/optimize",
        json={
            "strategy": "optimize_factor_exposure",
            "securities": [
                {"ticker": "IEFA", "weight": 20},
                {"ticker": "GLD", "weight": 20},
                {"ticker": "AGG", "weight": 20},
                {"ticker": "VEA", "weight": 20},
                {"ticker": "SPY", "weight": 20},
            ],
            "constraints": {"maximize_factor": "momentum"},
        },
    )

    assert response.status_code == 200
    body = response.json()
    current = body["factor_betas"]["current_portfolio"]
    optimized = body["factor_betas"]["optimized_portfolio"]

    assert optimized["momentum"] > current["momentum"]
