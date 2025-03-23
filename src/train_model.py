"""
Script to train, save, and evaluate a machine learning model.

This script performs the following steps:
1. Loads configuration from `config.json`.
2. Loads and splits the dataset into training and testing sets.
3. Processes the data for training and testing.
4. Trains a RandomForestClassifier model.
5. Saves the trained model, encoder, and label binarizer.
6. Runs inference on the test set.
7. Evaluates the model using precision, recall, and F1 score.
8. Evaluates model performance on data slices.
"""

import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    compute_model_metrics_on_slices_from_predictions,
    inference,
    save_model,
    train_model,
)
from utils import get_logger, load_config

# Initialize logger
logger = get_logger()


def prepare_data(
    data_path: Path, test_size: float, categorical_features: List[str], target_label: str, random_state: int
) -> Tuple[
    Tuple[pd.DataFrame, pd.DataFrame],
    Tuple[pd.DataFrame, pd.DataFrame, object, object],
    Tuple[pd.DataFrame, pd.DataFrame],
]:
    """
    Load and split the data, then process it for training and testing.

    Args:
        data_path (Path): Path to the dataset.
        test_size (float): Proportion of the dataset to include in
                           the test split.
        categorical_features (List[str]): List of categorical feature names.
        target_label (str): Name of the target label column.

    Returns:
        Tuple: Original dataset, train/test splits, processed training data,
               and processed testing data.
    """
    logger.info(f"Loading dataset from {data_path}...")
    data_df = pd.read_csv(data_path)
    logger.info("Dataset loaded successfully.")

    # Split data into training and testing sets
    logger.info(f"Splitting data into train and test sets" f"with test size {test_size}...")
    train, test = train_test_split(
        data_df, test_size=test_size, stratify=data_df[target_label], random_state=random_state
    )
    logger.info("Data split successfully.")

    # Process training data
    logger.info("Processing training data...")
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=categorical_features,
        label=target_label,
        training=True,
    )
    logger.info("Training data processed successfully.")

    # Process testing data
    logger.info("Processing testing data...")
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=categorical_features,
        label=target_label,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    logger.info("Testing data processed successfully.")

    return (
        (train, test),
        (X_train, y_train, encoder, lb),
        (X_test, y_test),
    )


def main() -> None:
    """
    Main function to train, save, and evaluate the machine learning model.
    """
    root_path = Path(os.getcwd())

    # Load configuration
    config_file = Path(root_path, "config.json")
    config = load_config(config_file)

    # Define paths
    data_path = Path(root_path, config["file_path"]["data"]["data_cls"])
    model_output_path = Path(root_path, config["file_path"]["model"]["model"])
    encoder_output_path = Path(root_path, config["file_path"]["model"]["encoder"])
    lb_output_path = Path(root_path, config["file_path"]["model"]["lb"])
    metrics_overall_output_path = Path(root_path, config["file_path"]["eval"]["metrics_overall"])
    metrics_sliced_output_path = Path(root_path, config["file_path"]["eval"]["metrics_sliced"])

    # Parameters
    test_size = config["train_test_model"]["test_size"]
    random_state = config["train_test_model"]["random_state"]
    categorical_features = config["train_test_model"]["categorical_features"]
    target_label = config["train_test_model"]["target_label"]
    slice_features = config["train_test_model"]["slice_features"]

    # Prepare data
    logger.info("Preparing data for training and testing...")
    (_, test), (X_train, y_train, encoder, lb), (X_test, y_test) = prepare_data(
        data_path, test_size, categorical_features, target_label, random_state
    )

    # Train the model
    logger.info("Training the model...")
    model = train_model(X_train, y_train, random_state=random_state)
    logger.info("Model trained successfully.")

    # Save the model, encoder, and label binarizer
    logger.info("Saving the model, encoder, and label binarizer...")
    save_model(
        model,
        encoder,
        lb,
        model_path=model_output_path,
        encoder_path=encoder_output_path,
        lb_path=lb_output_path,
    )
    logger.info("Model, encoder, and label binarizer saved successfully.")

    # Run inference
    logger.info("Running inference on the test set...")
    predictions = inference(model, X_test)
    logger.info("Inference completed successfully.")

    # Evaluate the model
    logger.info("Evaluating the model...")
    precision, recall, fbeta = compute_model_metrics(y_test, predictions, metrics_overall_output_path)
    logger.info(
        f"Model evaluation completed and saved sucessfully. Precision: {precision}, Recall: {recall}, F1 Score: {fbeta}"
    )

    # Evaluate Slice Performance
    logger.info("Evaluating slice performance...")
    compute_model_metrics_on_slices_from_predictions(
        X_df=test.drop(columns=[target_label]),
        y_true=y_test,
        y_pred=predictions,
        slice_features=slice_features,
        output_file=metrics_sliced_output_path,
    )
    logger.info("Slice performance evaluation completed and saved sucessfully.")


if __name__ == "__main__":
    main()
