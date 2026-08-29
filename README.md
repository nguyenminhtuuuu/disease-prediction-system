# Disease Prediction System Based on Blood Test Indicators

Python · Pandas · Scikit-learn · SHAP · Streamlit

## Overview

This project explores disease prediction based on blood test indicators,
with a focus on data quality analysis, class distribution, class imbalance,
predictive modeling, and model interpretability.

Two Kaggle datasets were merged into a unified dataset and analyzed before
training machine learning models for multi-class disease prediction.

## Dataset

- **Source:** Kaggle
- **Datasets:** 2
- **Records:** approximately 2,800
- **Classes:** 6 disease classes

## Data Analysis

Performed data quality checks and distribution analysis to examine:

- Missing values
- Outliers
- Class distribution
- Class imbalance

Two minority classes were identified among the six disease classes.

## Handling Class Imbalance

Selective SMOTE was applied using a custom sampling strategy to address
minority-class imbalance while preserving the majority-class distribution.

## Model

A Random Forest classifier was trained for multi-class disease prediction.

### Performance

**F1-score: 0.981**

> Specify whether this is Macro F1, Weighted F1, or another F1 metric
> based on the actual evaluation result before publishing.

## Model Interpretability

SHAP was used to identify the blood indicators that contributed most
strongly to model predictions.

This provided additional insight into model behavior and feature
importance.

## Application

An interactive Streamlit application was developed to allow users to
enter blood test indicators and receive a predicted disease result.

