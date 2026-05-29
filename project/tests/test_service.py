import pytest
from unittest.mock import patch

from src.service.app import app


def test_health_endpoint(client):
    """Test /health returns ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_request(client):
    """Test /predict with valid input."""
    payload = {
        "age": 30,
        "height_cm": 175,
        "weight_kg": 70,
        "sleep_hours": 7.5,
        "activity_index": 4.0,
        "smokes": "no",
        "gender": "F",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_fit"] in [0, 1]
    assert 0 <= data["probability"] <= 1


def test_predict_validation_error(client):
    """Test /predict rejects invalid input."""
    payload = {"age": 10}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
