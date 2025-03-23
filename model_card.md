# Model Card

## Model Details

- **Model name**: Random Forest Classifier for Census Income Prediction  
- **Model version**: v1.0  
- **Developed by**: [Laura Weber](https://github.com/LauRacoon2110)  
- **Model type**: Binary classification model  
- **Input**: Tabular data with demographic and employment features (both numeric and categorical)  
- **Output**: Binary label indicating whether an individual earns `>50K` or `<=50K`  
- **License**: [GPL-3.0](https://www.gnu.org/licenses/#GPL)  
- **Source code**: [udc_mldevops_cp3_ml_model_to_cap_with_fastapi](https://github.com/LauRacoon2110/udc_mldevops_cp3_ml_model_to_cap_with_fastapi)  

## Intended Use

This model is intended to assist in analyzing census-like datasets to predict whether an individual earns more than $50K per year based on demographic attributes. It is designed for:

- Educational and academic demonstrations  
- Exploratory data analysis  
- Benchmarking machine learning workflows  

**Not intended for**:

- Real-world deployment in hiring, loan approvals, or other high-stakes decisions  
- Use on real individuals without thorough fairness and compliance assessments  

## Training Data

- **Source**: [UCI Census Income Dataset](https://archive.ics.uci.edu/dataset/20/census+income)  
- **Instances**: ~48,842 rows  
- **Features**: 14 attributes, including age, education, occupation, etc.  
- **Target**: Income (`>50K` or `<=50K`)  

### Preprocessing Steps

- Whitespace stripping (columns + string values)  
- Replaced `?` with `NaN`  
- Dropped duplicates  
- Dropped rows with missing values (<1%)  
- Generated a profiling report using `ydata_profiling`
- Encoded categorical features with `OneHotEncoder`  
- Encoded target variable using `LabelBinarizer` 

## Evaluation Data

- **Split**: 80% train / 20% test  
- **Stratification**: Yes, by income category (`salaray`) 
- **Processing**: Same steps as training set using saved encoder & label binarizer  

## Metrics

**On the test set**:
- **Precision**: `0.7373`  
- **Recall**: `0.6460`  
- **F1 Score (F-beta with β=1)**: `0.6886`  
_Source: `model/overall_output.txt`_

### Example of Slice Metrics

| Feature (Slice)              | Precision | Recall | F1 Score |
|-----------------------------|-----------|--------|----------|
| Age = 39                    | 0.7451    | 0.7170 | 0.7308   |
| Age = 51                    | 0.8043    | 0.6852 | 0.7400   |
| Sex = Female                | 0.7835    | 0.5984 | 0.6786   |
| Education = Doctorate       | 0.8082    | 0.8676 | 0.8369   |
| Native Country = India      | 0.5714    | 1.0000 | 0.7273   |

_See `model/sliced_output.txt` for a full breakdown._

## Ethical Considerations

- Sensitive attributes (e.g., race, sex, marital status) may contribute to bias  
- No fairness-mitigation strategies have been applied  
- Model should not be used for decisions impacting real people without rigorous audits and ethical safeguards  

## Caveats and Recommendations

- Do **not** use this model in production without bias mitigation and fairness testing  
- Re-train periodically if population distribution changes  
- Explore fairness-aware modeling techniques for high-stakes applications  
