import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.service.app import app


@pytest.fixture
def client():
    """TestClient with mocked model dependencies."""
    with patch("src.service.app.model") as mock_model, patch(
        "src.service.app.preprocessor"
    ) as mock_prep:

        mock_prep.transform.return_value = [[0.1, 0.2, 0.3]]
        mock_model.predict.return_value = [1]
        mock_model.predict_proba.return_value = [[0.3, 0.7]]

        with TestClient(app) as c:
            yield c


def test_health_endpoint(client):
    """Test /health returns ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_valid_request(client):
    """Test /predict with valid input."""
    payload = {
        "age": 30,
        "weight_kg": 70,
        "heart_rate": 65,
        "sleep_hours": 7.5,
        "activity_days_week": 5,
        "smoking": "no",
        "nutrition_quality": "high",
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
