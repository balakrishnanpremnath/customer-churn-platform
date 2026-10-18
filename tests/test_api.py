
import pytest
from fastapi.testclient import TestClient

from src.churn_platform.api import app


CUSTOMER = {
    "frequency_180d": 12,
    "monetary_180d": 6000,
    "quantity_180d": 40,
    "active_days_180d": 10,
    "unique_products_180d": 15,
    "recency_days": 20,
    "tenure_180d": 150,
    "avg_order_value_180d": 500,
    "orders_30d": 2,
    "spend_30d": 1000,
    "orders_90d": 6,
    "spend_90d": 3000,
    "spend_share_30d": 0.1667,
    "has_repeat_purchase_day_180d": 1,
    "avg_repeat_gap_days_180d": 14,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
    }


def test_model_info_endpoint(client):
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert data["model"] == "frozen_random_forest"
    assert data["feature_count"] == 15
    assert data["training_snapshots"] == 14987
    assert data["threshold"] == 0.5
    assert data["probabilities_calibrated"] is False


def test_prediction_endpoint(client):
    first = client.post("/predict", json=CUSTOMER)
    second = client.post("/predict", json=CUSTOMER)

    assert first.status_code == 200
    assert second.status_code == 200

    result = first.json()

    score = result["inactivity_risk_score"]

    assert 0 <= score <= 1
    assert result["threshold"] == 0.5
    assert result["prediction_horizon_days"] == 90

    assert result["predicted_inactive"] == (
        score >= result["threshold"]
    )

    # Same input should produce the same prediction.
    assert first.json() == second.json()


def test_invalid_customer_inputs(client):
    missing = CUSTOMER.copy()
    missing.pop("recency_days")

    response = client.post("/predict", json=missing)
    assert response.status_code == 422

    negative = CUSTOMER.copy()
    negative["monetary_180d"] = -100

    response = client.post("/predict", json=negative)
    assert response.status_code == 422

    extra = CUSTOMER.copy()
    extra["unexpected_feature"] = 123

    response = client.post("/predict", json=extra)
    assert response.status_code == 422
