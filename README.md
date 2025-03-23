# Salary Prediction API with ML & FastAPI

A full ML Ops-capable project that trains and deploys a **Random Forest Classifier** to predict whether an individual's annual income exceeds $50K based on U.S. Census data. The pipeline includes data preprocessing, model training, evaluation, slice-based performance analysis, and deployment via a FastAPI-based REST API.

---

## Project Structure

```
.
├── data/                    # Cleaned data & profiling report
├── model/                   # Trained model, encoder, and output metrics
├── screenshots/             # Screenshots for documentation
├── src/                     # Core Modules to clean data, train the model and corresponding utils
│   ├── ml/                  # Core ML modules (data prep, model logic)
├── tests/                   # Unit tests
├── .pre-commit-config.yaml  # Linting and formatting automation
├── config.json              # Centralized project config
├── main.py                  # FastAPI application
├── model_card.md            # Model Card with performance metrics
├── requirements.txt         # Required Python packages
├── sanitycheck.py           # Script to validate rubric requirements are met
└── README.md                # This file
```

---

## Model Overview

- **Algorithm**: RandomForestClassifier (`scikit-learn`)
- **Dataset**: [UCI Census Income Dataset](https://archive.ics.uci.edu/dataset/20/census+income)
- **Target**: Binary classification (`>50K` / `<=50K`)
- **Features**: Age, education, marital status, occupation, etc.
- **Preprocessing**: Missing value handling, encoding, profiling

### Metrics (Overall)
- **Precision**: `0.7373`  
- **Recall**: `0.6460`  
- **F1 Score (F-beta with β=1)**: `0.6886` 

### Slice Metrics
Model performance was also evaluated across slices of categorical features like `age`, `education`, `sex`, and `occupation`.  
See: [`model/sliced_output.txt`](./model/sliced_output.txt)

---

## API Endpoints

| Endpoint       | Method | Description                        |
|----------------|--------|------------------------------------|
| `/`            | GET    | Welcome message                    |
| `/predict`     | POST   | Predict income class from features |

Docs: visit [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI.


---

## Installation

### 1. Python Version

```bash
python --version  # Expected: Python 3.8+
```

If you're using `pyenv`, your `.python-version` is already set.

---

### 2. Create and Activate a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

---

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

---

## Run the Code

### Clean the data
_(done only once in this experimental setting and stored in `data/census_cls.csv`)_

```bash
python src/preprocess_data.py
```

### Train the Model

```bash
python src/train_model.py
```
---

## Testing

Unit and integration tests are located in the `src/tests/` directory.

```bash
pytest
```

---

## Code Quality & Linting

This project uses the following tools:

- [`flake8`](https://flake8.pycqa.org/) – code style and linting
- [`black`](https://black.readthedocs.io/) – automatic code formatting
- [`isort`](https://pycqa.github.io/isort/) – import sorting
- [`mypy`](https://mypy.readthedocs.io/) – static type checking
- [`pytest`](https://docs.pytest.org/) – testing framework
- [`pre-commit`](https://pre-commit.com/) – run all of the above on each commit

### Enable pre-commit hooks:

```bash
pre-commit install
pre-commit run --all-files  # optional: run manually
```
---

## Run the APP
### Run the FastAPI server

```bash
uvicorn main:app --reload
```

Then go to: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📄 Model Card

Details on the model training process, intended use, metrics, and ethical considerations are available in:

👉 [`model_card.md`](./model_card.md)

---

## 📝 License

[GPL-3.0](https://www.gnu.org/licenses/#GPL)  

---
