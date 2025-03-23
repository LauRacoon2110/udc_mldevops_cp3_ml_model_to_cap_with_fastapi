"""
Module for preprocessing the Census dataset.

This module performs the following steps:
1. Loads the raw dataset.
2. Cleans the dataset by:
   - Stripping leading and trailing whitespaces from column names and string values.
   - Replacing '?' with NaN.
   - Dropping duplicate rows.
   - Dropping rows with missing values (if missing values are <1%).
3. Saves the cleaned dataset to a CSV file.
4. Generates a profiling report for the cleaned dataset.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport

from utils import get_logger, load_config

# Initialize logger
logger = get_logger()


def preprocess_data(input_path: Path, output_path: Path, report_path: Path) -> None:
    """
    Preprocess the Census dataset and save the cleaned dataset.

    Args:
        input_path (Path): Path to the raw dataset (CSV file).
        output_path (Path): Path to save the cleaned dataset (CSV file).
        report_path (Path): Path to save the profiling report (HTML file).
    """
    logger.info(f"Loading dataset from {input_path}...")
    census_raw_df = pd.read_csv(input_path)
    logger.info("Dataset loaded successfully.")

    # Create a copy of the dataset for cleaning
    census_cls_df = census_raw_df.copy()

    # Strip leading and trailing whitespaces from column names
    logger.info("Stripping whitespaces from column names...")
    census_cls_df.columns = census_cls_df.columns.str.strip()

    # Strip whitespaces from string values
    logger.info("Stripping whitespaces from string values...")
    census_cls_df = census_cls_df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Replace '?' with NaN
    logger.info("Replacing '?' with NaN...")
    census_cls_df.replace("?", np.nan, inplace=True)

    # Drop duplicate rows
    logger.info("Dropping duplicate rows...")
    census_cls_df = census_cls_df.drop_duplicates()

    # Drop rows with missing values (if missing values are <1%)
    missing_percentage = census_cls_df.isnull().mean().max() * 100
    if missing_percentage < 1:
        logger.info("Dropping rows with missing values (less than 1% missing)...")
        census_cls_df = census_cls_df.dropna()
    else:
        logger.warning(f"Dataset contains {missing_percentage:.2f}% missing values. Consider handling them.")

    # Save the cleaned dataset to a CSV file
    logger.info(f"Saving cleaned dataset to {output_path}...")
    census_cls_df.to_csv(output_path, index=False)
    logger.info("Cleaned dataset saved successfully.")

    # Generate a profiling report
    logger.info("Generating profiling report...")
    profile = ProfileReport(
        census_cls_df,
        title="Census Data Profiling Report",
        explorative=True,
        minimal=False,
    )
    profile.to_file(report_path)
    logger.info(f"Profiling report saved to {report_path}.")


if __name__ == "__main__":
    # Load configuration
    root_path = Path(os.getcwd())
    config_file = Path(root_path, "config.json")
    config = load_config(config_file)

    # Define paths from the configuration file
    input_file = Path(root_path, config["file_path"]["data"]["data_raw"])
    output_file = Path(root_path, config["file_path"]["data"]["data_cls"])
    report_file = Path(root_path, config["file_path"]["eval"]["profiling_report"])

    # Run preprocessing
    preprocess_data(input_file, output_file, report_file)
