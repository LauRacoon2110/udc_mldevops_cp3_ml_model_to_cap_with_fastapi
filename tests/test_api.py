from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


@pytest.fixture
def sample_input():
    """Fixture to provide sample input for predictions."""
    return [
        {
            "age": 39,
            "workclass": "State-gov",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        }
    ]


def test_welcome():
    """Test the GET / endpoint returns the welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to our Salary Prediction API"}


@patch("main.process_data", return_value=(np.random.rand(1, 108), None, None, None))
@patch("main.inference", return_value=[1])
def test_post_predict_over_50k(mock_inference, mock_process, sample_input):
    """Test POST /predict where prediction is >50K"""
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    result = response.json()
    assert result["numerical_predictions"] == [1]
    assert result["mapped_predictions"] == [">50K"]
    mock_inference.assert_called_once()
    mock_process.assert_called_once()


@patch("main.process_data", return_value=(np.random.rand(1, 108), None, None, None))
@patch("main.inference", return_value=[0])
def test_post_predict_under_50k(mock_inference, mock_process, sample_input):
    """Test POST /predict where prediction is <=50K"""
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 200
    result = response.json()
    assert result["numerical_predictions"] == [0]
    assert result["mapped_predictions"] == ["<=50K"]
    mock_inference.assert_called_once()
    mock_process.assert_called_once()


def test_invalid_endpoint():
    """Test accessing an invalid endpoint returns 404."""
    response = client.get("/invalid-endpoint")
    assert response.status_code == 404


def test_post_predict_missing_fields():
    """Test POST /predict with missing fields in the input."""
    incomplete_input = [
        {
            "age": 39,
            "workclass": "State-gov",
            # Missing other required fields
        }
    ]
    response = client.post("/predict", json=incomplete_input)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_post_predict_empty_input():
    """Test POST /predict with an empty input list."""
    response = client.post("/predict", json=[])
    assert response.status_code == 422
    assert response.json() == {"detail": "Input data cannot be empty"}


def test_post_predict_invalid_data_types():
    """Test POST /predict with invalid data types."""
    invalid_input = [
        {
            "age": "thirty-nine",  # Invalid type
            "workclass": "State-gov",
            "fnlgt": "seventy-seven thousand",  # Invalid type
            "education": "Bachelors",
            "education-num": "thirteen",  # Invalid type
            "marital-status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital-gain": "two thousand",  # Invalid type
            "capital-loss": 0,
            "hours-per-week": "forty",  # Invalid type
            "native-country": "United-States",
        }
    ]
    response = client.post("/predict", json=invalid_input)
    assert response.status_code == 422
    assert "detail" in response.json()


@patch("main.process_data", return_value=(np.random.rand(2, 108), None, None, None))
@patch("main.inference", return_value=[1, 0])
def test_post_predict_batch(mock_inference, mock_process, sample_input):
    """Test POST /predict with multiple input records."""
    batch_input = sample_input * 2  # Create two records
    response = client.post("/predict", json=batch_input)
    assert response.status_code == 200
    result = response.json()
    assert result["numerical_predictions"] == [1, 0]
    assert result["mapped_predictions"] == [">50K", "<=50K"]
    mock_inference.assert_called_once()
    mock_process.assert_called_once()


@patch("main.process_data", return_value=(np.random.rand(1, 108), None, None, None))
@patch("main.inference", side_effect=Exception("Unexpected error"))
def test_post_predict_unexpected_error(mock_inference, mock_process, sample_input):
    """Test POST /predict raises an unexpected error during inference."""
    response = client.post("/predict", json=sample_input)
    assert response.status_code == 500
    assert "detail" in response.json()
    assert response.json()["detail"] == "Error during model inference: Unexpected error"
    mock_inference.assert_called_once()
    mock_process.assert_called_once()
