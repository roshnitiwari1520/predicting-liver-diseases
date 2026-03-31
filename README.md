# Liver Disease Prediction Model

## Problem Statement
Liver disease affects millions globally and is often diagnosed late. 
This project builds a machine learning model to predict liver disease 
from patient medical data, enabling early detection.

## Dataset
- Source: [ILPD Dataset - UCI Machine Learning Repository]
- 583 patient records, 10 features
- Target: Binary classification (liver disease / no disease)

## Approach
1. Exploratory Data Analysis — identified class imbalance, outliers
2. Preprocessing — handled missing values, scaled features
3. Model Building — compared Logistic Regression vs Random Forest
4. Evaluation — selected Random Forest based on F1 score

## Results
| Model | Accuracy | F1 Score |
|-------|----------|----------|
| Logistic Regression | 71% | 0.68 |
| Random Forest | 85% | 0.83 |

## Key Findings
- Age and gender are strong predictors
- Albumin and protein levels are the most important features
- Class imbalance was the biggest challenge

## Tech Stack
Python, Pandas, Scikit-learn, Matplotlib, Seaborn

## How to Run
```bash
git clone https://github.com/roshnitiwari1520/predicting-liver-diseases
pip install -r requirements.txt
jupyter notebook liver_disease_prediction.ipynb
```
