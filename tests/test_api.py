# tests/test_api.py
import requests
import pytest

# The base URL for your running FastAPI app
BASE_URL = "http://localhost:8000"

def test_root_endpoint():
    """Tests if the root endpoint is reachable."""
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Iris Classifier API!"}

def test_prediction_endpoint_success():
    """Tests a successful prediction."""
    # This is a sample input for Iris setosa (class 0)
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = requests.post(f"{BASE_URL}/predict", json=payload)

    assert response.status_code == 200
    response_json = response.json()
    assert "prediction" in response_json
    assert isinstance(response_json["prediction"], int)
    # We expect this specific input to predict class 0
    assert response_json["prediction"] == 0

def test_prediction_endpoint_invalid_data():
    """Tests the endpoint's response to invalid data (missing field)."""
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
        # Missing "petal_width"
    }

    response = requests.post(f"{BASE_URL}/predict", json=payload)

    # FastAPI should return a 422 Unprocessable Entity error
    assert response.status_code == 422