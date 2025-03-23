from typing import Any, Dict, List

import requests


def get_prediction(payload: List[Dict[str, Any]], url: str) -> None:
    """
    Sends a POST request to the specified API endpoint with the given payload
    and prints the status code and response.

    Args:
        payload (List[Dict[str, Any]]): The input data for the prediction API.
        url (str): The API endpoint URL.

    Returns:
        None
    """
    try:
        # Send the POST request
        response = requests.post(url, json=payload)

        # Print the status code
        print(f"Status Code: {response.status_code}")

        # Print the response JSON or raw text if JSON parsing fails
        try:
            print("Response JSON:", response.json())
        except ValueError:
            print("Response is not valid JSON:", response.text)

    except requests.RequestException as e:
        # Handle any request-related exceptions
        print(f"An error occurred while making the request: {e}")


if __name__ == "__main__":
    # Define the payload
    payload = [
        {
            "age": 40,
            "workclass": "Private",
            "fnlgt": 77516,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Divorced",
            "occupation": "Sales",
            "relationship": "Unmarried",
            "race": "Black",
            "sex": "Female",
            "capital-gain": 2174,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States",
        }
    ]

    # Define the API endpoint
    api_url = "https://udc-mldevops-cp3-ml-model-to-cap-with.onrender.com/predict"

    # Call the function to get predictions
    get_prediction(payload, api_url)
