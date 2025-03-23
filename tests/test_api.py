import tempfile
from typing import Generator, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.ml.model import (
    compute_model_metrics,
    compute_model_metrics_on_slices_from_predictions,
    inference,
    load_model,
    save_model,
    train_model,
)


@pytest.fixture
def sample_data() -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Fixture to provide sample training data."""
    X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y_train = np.array([0, 1, 0])
    yield X_train, y_train


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Fixture to provide a sample DataFrame for slicing."""
    data = {
        "feature1": ["A", "A", "B", "B"],
        "feature2": ["X", "Y", "X", "Y"],
        "label": [1, 0, 1, 0],
    }
    return pd.DataFrame(data)


def test_train_model(sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test the train_model function."""
    X_train, y_train = sample_data
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)
    assert model.n_classes_ == 2  # Ensure the model was trained on binary labels


@patch("src.ml.model.joblib.dump")
def test_save_model(mock_joblib_dump: MagicMock) -> None:
    """Test the save_model function."""
    model = MagicMock()
    encoder = MagicMock()
    lb = MagicMock()
    save_model(model, encoder, lb, "model.pkl", "encoder.pkl", "lb.pkl")
    assert mock_joblib_dump.call_count == 3  # Ensure all three objects were saved


@patch("src.ml.model.joblib.load")
def test_load_model(mock_joblib_load: MagicMock) -> None:
    """Test the load_model function."""
    mock_joblib_load.side_effect = [MagicMock(), MagicMock(), MagicMock()]
    model, encoder, lb = load_model("model.pkl", "encoder.pkl", "lb.pkl")
    assert mock_joblib_load.call_count == 3  # Ensure all three objects were loaded
    assert model is not None
    assert encoder is not None
    assert lb is not None


def test_inference(sample_data: Tuple[np.ndarray, np.ndarray]) -> None:
    """Test the inference function."""
    X_train, y_train = sample_data
    model = train_model(X_train, y_train)
    predictions = inference(model, X_train)
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(X_train)


def test_compute_model_metrics() -> None:
    """Test the compute_model_metrics function."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        precision, recall, f1 = compute_model_metrics(y_true, y_pred, output_file=temp_file.name)
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1


def test_compute_model_metrics_on_slices_from_predictions(sample_dataframe: pd.DataFrame) -> None:
    """Test the compute_model_metrics_on_slices_from_predictions function."""
    df = sample_dataframe
    y_true = df["label"].values
    y_pred = np.array([1, 0, 1, 1])  # Example predictions
    slice_features = ["feature1", "feature2"]
    with tempfile.NamedTemporaryFile(delete=True) as temp_file:
        results = compute_model_metrics_on_slices_from_predictions(
            X_df=df,
            y_true=y_true,
            y_pred=y_pred,
            slice_features=slice_features,
            output_file=temp_file.name,
        )
        assert isinstance(results, list)
        assert len(results) > 0
        for result in results:
            assert len(result) == 5  # (feature, value, precision, recall, f1)
