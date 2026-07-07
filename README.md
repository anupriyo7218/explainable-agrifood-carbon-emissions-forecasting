# Explainable Machine Learning for Forecasting Agri-food Carbon Emissions

## Overview

This project develops an explainable machine learning pipeline for forecasting agri-food carbon emissions using publicly available FAO/IPCC environmental data.

The pipeline focuses on producing reproducible carbon-emission predictions while maintaining model transparency through SHAP (SHapley Additive exPlanations). The project follows a modular workflow consisting of data preprocessing, model training, model evaluation, and explainability.

---

## Dataset

**Dataset:** Agrofood_co2_emission.csv

The dataset contains environmental and demographic indicators used to predict total agri-food carbon emissions.

---

## Project Structure

```
data/
├── raw/
├── processed/

models/

reports/
├── predictions/
└── shap/

data_preprocessing.py
model_training.py
shap_analysis.py
requirements.txt
README.md
```

---

## Workflow

### 1. Data Preprocessing

The preprocessing pipeline performs:

- Dataset validation
- Duplicate removal
- Missing value handling
- Leakage prevention
- Feature selection
- One-hot encoding
- Temporal train-test split
- Processed dataset generation

---

### 2. Model Training

Three ensemble regression models were trained and evaluated:

- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

Evaluation metrics:

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

The best-performing model is selected automatically based on R².

---

### 3. Model Explainability

The selected Gradient Boosting model is interpreted using SHAP (SHapley Additive exPlanations).

Generated outputs include:

- SHAP Summary (Beeswarm) Plot
- SHAP Feature Importance Plot
- Raw SHAP Values
- Feature Order File

---

## Model Performance

| Model | R² | MAE | RMSE |
|------|------:|------:|------:|
| Random Forest | 0.9657 | 17009.09 | 55254.68 |
| **Gradient Boosting** | **0.9686** | 22506.75 | 52855.54 |
| XGBoost | 0.9274 | 20725.07 | 80369.24 |

---

## Generated Outputs

The repository includes:

- Processed datasets
- Trained machine learning models
- Model comparison report
- Prediction outputs
- SHAP explainability results

---

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run preprocessing:

```bash
python data_preprocessing.py
```

Train models:

```bash
python model_training.py
```

Generate SHAP explanations:

```bash
python shap_analysis.py
```

---

## Repository Note

The file `randomforest.pkl` exceeds GitHub's 25 MB upload limit and has therefore been archived as `randomforest.rar`.

The archive is fully downloadable and can be extracted normally. No password is required.


---

## Author

Anupriyo Chakraborty
