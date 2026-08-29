# Disease Prediction System Based on Blood Test Indicators
**Tools:** Python, Pandas, Scikit-learn, SMOTE, Imbalanced-learn, Streamlit

## 1. Introduction
The dataset consists of blood sample records containing various blood indicators such as Glucose, Cholesterol, Hemoglobin, etc., collected to predict 6 different health conditions. 

The primary goal of this project is to build a robust Machine Learning model capable of multi-class disease prediction while addressing severe class imbalance in the medical dataset using SMOTE. 

## 2. Dataset Overview
The unified dataset contains **2,837 records** and **25 columns** (24 features and 1 target variable).

| Field Name | Data Type | Description |
| :--- | :--- | :--- |
| `Glucose` | FLOAT | Blood sugar level |
| `Cholesterol` | FLOAT | Cholesterol level |
| `Hemoglobin` | FLOAT | Hemoglobin concentration |
| `Platelets` | FLOAT | Platelet count |
| `White Blood Cells` | FLOAT | White blood cell count |
| `Disease` (Target) | STRING | 6 classes: *Khỏe mạnh, Tiểu đường, Tan máu bẩm sinh, Thiếu máu, Huyết khối, Tim mạch* |

## 3. Data Processing & Exploratory Data Analysis (EDA)

Before training, the categorical labels were translated into Vietnamese for consistency:

```python
# Chuyển label sang tiếng Việt
df_blood['Disease'] = df_blood['Disease'].replace({
    'Healthy': 'Khỏe mạnh',
    'Diabetes' : 'Tiểu đường',
    'Thalasse' : 'Tan máu bẩm sinh',
    'Anemia' : 'Thiếu máu',
    'Thromboc' : 'Huyết khối',
    'Heart Di' : 'Tim mạch'
})
