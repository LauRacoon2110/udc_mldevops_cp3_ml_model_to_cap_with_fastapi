"""
Module for training, saving, loading, and evaluating a machine learning model.

This module provides the following functionalities:
1. Train a RandomForestClassifier model.
2. Save the trained model, encoder, and label binarizer to disk.
3. Load the model, encoder, and label binarizer from disk.
4. Perform inference using the trained model.
5. Compute evaluation metrics (precision, recall, F1 score) for the model.
6. Compute metrics on slices of data using precomputed predictions.
"""

from typing import List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score


def train_model(
    X_train: NDArray[np.float_],
    y_train: NDArray[np.int_],
    random_state: Optional[int] = 42,
) -> RandomForestClassifier:
    """
    Train a RandomForestClassifier model.

    Args:
        X_train (NDArray[np.float_]): Training data as a NumPy array of floats.
        y_train (NDArray[np.int_]): Labels for the training data as a NumPy array of integers.
        random_state (Optional[int]): Random state for reproducibility. Defaults to 42.

    Returns:
        RandomForestClassifier: A trained RandomForestClassifier model.
    """
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def save_model(
    model: RandomForestClassifier,
    encoder: object,
    lb: object,
    model_path: str,
    encoder_path: str,
    lb_path: str,
) -> None:
    """
    Save the trained model, encoder, and label binarizer to disk.

    Args:
        model (RandomForestClassifier): Trained machine learning model.
        encoder (object): Trained encoder.
        lb (object): Trained label binarizer.
        model_path (str): Path to save the model.
        encoder_path (str): Path to save the encoder.
        lb_path (str): Path to save the label binarizer.
    """
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)
    joblib.dump(lb, lb_path)


def load_model(model_path: str, encoder_path: str, lb_path: str) -> Tuple[RandomForestClassifier, object, object]:
    """
    Load the trained model, encoder, and label binarizer from disk.

    Args:
        model_path (str): Path to load the model.
        encoder_path (str): Path to load the encoder.
        lb_path (str): Path to load the label binarizer.

    Returns:
        Tuple[RandomForestClassifier, object, object]: Loaded model, encoder, and label binarizer.
    """
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    lb = joblib.load(lb_path)
    return model, encoder, lb


def inference(model: RandomForestClassifier, X: NDArray[np.float_]) -> NDArray[np.int_]:
    """
    Perform inference using the trained model.

    Args:
        model (RandomForestClassifier): Trained machine learning model.
        X (NDArray[np.float_]): Data used for prediction as a NumPy array of floats.

    Returns:
        NDArray[np.int_]: Predictions from the model as a NumPy array of integers.
    """
    predictions = model.predict(X)
    return np.asarray(predictions, dtype=np.int_)  # Explicitly cast to NDArray[np.int_]


def compute_model_metrics(
    y: NDArray[np.int_],
    preds: NDArray[np.int_],
    output_file: Optional[str] = "overall_output.txt",
) -> Tuple[float, float, float]:
    """
    Compute precision, recall, and F1 score for the model and write the results to a file.

    Args:
        y (NDArray[np.int_]): Known labels as a NumPy array of integers.
        preds (NDArray[np.int_]): Predicted labels as a NumPy array of integers.
        output_file (Optional[str]): Path to the file where metrics will be written. Defaults to "overall_output.txt".

    Returns:
        Tuple[float, float, float]: Precision, recall, and F1 score.
    """
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)

    if output_file:
        with open(output_file, "w") as f:
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {fbeta:.4f}\n")

    return precision, recall, fbeta


def compute_model_metrics_on_slices_from_predictions(
    X_df: pd.DataFrame,
    y_true: NDArray[np.int_],
    y_pred: NDArray[np.int_],
    slice_features: List[str],
    output_file: str = "slice_output.txt",
) -> List[Tuple[str, str, float, float, float]]:
    """
    Compute model metrics on data slices using precomputed predictions.

    Args:
        X_df (pd.DataFrame): Original (pre-encoded) features DataFrame.
        y_true (NDArray[np.int_]): Ground truth labels as a NumPy array of integers.
        y_pred (NDArray[np.int_]): Model predictions as a NumPy array of integers.
        slice_features (List[str]): Feature(s) to slice by.
        output_file (str): Path to the file where slice metrics will be written. Defaults to "slice_output.txt".

    Returns:
        List[Tuple[str, str, float, float, float]]: Metrics for each slice
                                                   (feature, value, precision, recall, F1 score).
    """
    results = []

    for feature in slice_features:
        for value in X_df[feature].unique():
            mask = X_df[feature] == value
            y_true_slice = y_true[mask]
            y_pred_slice = y_pred[mask]

            if len(y_true_slice) == 0:
                continue

            precision = precision_score(y_true_slice, y_pred_slice, zero_division=1)
            recall = recall_score(y_true_slice, y_pred_slice, zero_division=1)
            fbeta = fbeta_score(y_true_slice, y_pred_slice, beta=1, zero_division=1)

            results.append((feature, value, precision, recall, fbeta))

    with open(output_file, "w") as file:
        for feature, value, p, r, f in results:
            file.write(f"{feature}={value} -> precision: {p:.4f}, recall: {r:.4f}, fbeta: {f:.4f}\n")

    return results
