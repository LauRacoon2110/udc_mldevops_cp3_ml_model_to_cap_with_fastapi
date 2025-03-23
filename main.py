"""
Main module for the Salary Prediction API.

This API provides the following endpoints:
1. `GET /` - A welcome message.
2. `POST /predict` - Accepts input data, processes it, performs inference,
   and returns predictions.

The API uses a pre-trained RandomForestClassifier model along with an encoder
and label binarizer for preprocessing.
"""

import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.ml.data import process_data  # type: ignore
from src.ml.model import inference, load_model
from src.utils import load_config

# Initialize FastAPI app
app = FastAPI()

# Load configuration
root_path = Path(os.getcwd())
config_file = Path(root_path, "config.json")
config = load_config(str(config_file))

# Define paths
model_output_path = Path(root_path, config["file_path"]["model"]["model"])
encoder_output_path = Path(root_path, config["file_path"]["model"]["encoder"])
lb_output_path = Path(root_path, config["file_path"]["model"]["lb"])

# Load model, encoder, and label binarizer
model, encoder, lb = load_model(str(model_output_path), str(encoder_output_path), str(lb_output_path))

# Define categorical features
categorical_features = config["train_test_model"]["categorical_features"]


class PredictionRequest(BaseModel):
    """
    Pydantic model for validating and parsing input data for predictions.
    """

    age: int = Field(example=40)
    workclass: str = Field(example="Private")
    fnlgt: int = Field(example=77516)
    education: str = Field(example="Bachelors")
    education_num: int = Field(alias="education-num", example=13)
    marital_status: str = Field(alias="marital-status", example="Divorced")
    occupation: str = Field(example="Sales")
    relationship: str = Field(example="Unmarried")
    race: str = Field(example="Black")
    sex: str = Field(example="Female")
    capital_gain: int = Field(alias="capital-gain", example=2174)
    capital_loss: int = Field(alias="capital-loss", example=0)
    hours_per_week: int = Field(alias="hours-per-week", example=40)
    native_country: str = Field(alias="native-country", example="United-States")

    class Config:
        allow_population_by_field_name = True


class PredictionResponse(BaseModel):
    numerical_predictions: List[int]
    mapped_predictions: List[str]


@app.get("/")
async def welcome() -> Dict[str, str]:
    """
    Welcome endpoint for the API.

    Returns:
        Dict[str, str]: A welcome message.
    """
    return {"message": "Welcome to our Salary Prediction API"}


@app.post("/predict")
async def predict(
    data: List[PredictionRequest],
) -> PredictionResponse:
    """
    Endpoint to handle predictions.

    Args:
        data (List[PredictionRequest]): List of input data for prediction.

    Returns:
        Dict[str, List[Any]]: Predictions in both numerical
                              and human-readable formats.
    """
    if not data:
        raise HTTPException(status_code=422, detail="Input data cannot be empty")

    try:
        # Convert input data to a DataFrame
        input_data = [item.dict(by_alias=True) for item in data]
        input_df = pd.DataFrame(input_data)

        # Preprocess the input data
        try:
            X, _, _, _ = process_data(
                X=input_df,
                categorical_features=categorical_features,
                label=None,
                training=False,
                encoder=encoder,
                lb=lb,
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error during data preprocessing: {str(e)}",
            )

        # Perform inference
        try:
            predictions = inference(model, X)
            predictions_list = [int(p) for p in predictions]
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error during model inference: {str(e)}",
            )

        # Map numerical predictions to human-readable values
        pred_dict = {1: ">50K", 0: "<=50K"}
        mapped_predictions = [pred_dict[p] for p in predictions_list]

        return PredictionResponse(
            numerical_predictions=predictions_list,
            mapped_predictions=mapped_predictions,
        )

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions to return appropriate responses
        raise http_exc
    except Exception as e:
        # Catch any unexpected errors
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}",
        )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
