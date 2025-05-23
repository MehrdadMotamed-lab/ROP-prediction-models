# Predicting Treatment-Needed Retinopathy of Prematurity (ROP)

This project aims to develop machine learning models to predict the need for treatment (either intravitreal bevacizumab or laser photocoagulation) to complete retinal vascularization in neonates with Retinopathy of Prematurity (ROP).

## 🧠 Objective

Predict whether an ROP patient will require treatment or complete vascularization without intervention using clinical and demographic data.

- **Label**: `event1`  
  - `1`: Treatment required (IVB or laser)
  - `0`: Spontaneous vascularization (no treatment)

## 📊 Dataset

The dataset includes the following features selected based on domain expertise:

| Feature            | Type       |
|--------------------|------------|
| Sex                | Binary     |
| Mother_Education   | Ordinal    |
| Father_Education   | Ordinal    |
| CPR                | Binary     |
| low_Apgar          | Binary     |
| Birth_Age          | Continuous |
| Birth_Weight       | Continuous |
| NICU_Time          | Continuous |
| Age                | Continuous |
| Weight             | Continuous |
| ROP_Stage          | Ordinal    |
| ROP_Zone           | Ordinal    |
| Plus               | Binary (1=plus, 2=no plus) |

## 🔍 Exploratory Data Analysis (EDA)

- Summary statistics
- Distribution plots of continuous features
- Bar plots for categorical features
- Pairplots
- Correlation heatmaps

## 🧹 Preprocessing

- Handling missing values
- Feature scaling
- Balancing using **ADASYN (Adaptive Synthetic Sampling)** to address class imbalance

## 🧪 Machine Learning Models

The following classifiers were evaluated:

| Classifier              | Highlights                                   |
|-------------------------|----------------------------------------------|
| Logistic Regression     | Baseline linear model                        |
| Decision Tree           | Non-linear, interpretable model              |
| K-Nearest Neighbors     | Lazy learner based on distance               |
| Naive Bayes             | Probabilistic, efficient for high-dim data   |
| Support Vector Machine  | Margin-based classifier                      |
| XGBoost                 | High-performing gradient boosting            |
| Artificial Neural Network (ANN) | Deep learning model with hidden layers |

Each model is evaluated using:
- Confusion Matrix
- Precision, Recall, F1-score
- ROC-AUC

## 🌳 Feature Importance

- Feature importance calculated using a Random Forest classifier.

## 🧠 Model Explainability

SHAP (SHapley Additive exPlanations) values were computed to explain model predictions and determine the most influential features contributing to each outcome.

## 📁 Repository Contents

- `ROP_ADASYN-SHAP.ipynb`: Full pipeline from preprocessing to explainability
- `DataSet-ROP.csv`: Input data (not included in this repository for privacy)


## 📦 Requirements

Install necessary packages using:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib xgboost shap
