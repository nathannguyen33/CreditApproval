# Credit Approval Prediction Project

### Nathan Nguyen & Charlie Nguyen  
**Course**: CS 171  
**Date**: 2025

---

## Project Overview

This project applies machine learning to predict whether a credit application should be **approved or denied**, using a dataset of 690 anonymized applicant records. The motivation behind this project is to improve **fairness, transparency, and consistency** in credit evaluation decisions, with particular attention to addressing **biases** and **class imbalances** in the data.

---

## Dataset Description

- **Source**: [UCI Credit Screening Dataset](https://archive.ics.uci.edu/ml/datasets/credit+approval)
- **Records**: 690
- **Features**: 15 attributes (A1–A15) + 1 target variable (A16)
- **Target (A16)**:
  - `1` = Approved  
  - `0` = Denied
- **Feature Types**:
  - **Categorical**: A1, A4, A5, A6, A7, A9, A10, A12, A13  
  - **Numerical**: A2, A3, A8, A11, A14, A15

---

## Data Preprocessing

- Imputed missing values (`?`) using:
  - **Mode** for categorical columns
  - **Median** for numerical columns
- Applied **Label Encoding** to categorical features
- Used **Min-Max Normalization** for numerical values
- Converted class labels (`+`, `-`) to binary format: `1` and `0`
- Removed outliers using **Z-score filtering**

---

## Exploratory Analysis

- Visualized class imbalance using bar plots
- Investigated skewness and feature dominance
- Assessed correlations using a heatmap (see *page 8*)
- Identified potential **demographic bias** in feature A1

---

## Models & Techniques

### Baseline Models:
- **Logistic Regression** – interpretable and probabilistic
- **Decision Tree** – captures non-linear relationships
- **K-Nearest Neighbors** – useful for similarity-based classification

### Advanced Models:
- **Random Forest**
- **Support Vector Machine (SVM)**
- **Gradient Boosting**

### Evaluation Metrics:
- Accuracy, Precision, Recall, F1-score
- Confusion Matrix
- ROC Curve and AUC Scores (see *page 17*)

### Feature Selection:
- Used `SelectKBest` with ANOVA F-test
- Selected top 10 features for final modeling

---

## Addressing Class Imbalance

- Resampling: Oversampling/Undersampling
- Class weight adjustments
- Threshold tuning to optimize minority class recall

---

## Libraries & Tools

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn` for modeling, preprocessing, and metrics
- Google Colab for development environment

---

## Results Summary

| Model                 | Accuracy | AUC Score |
|----------------------|----------|-----------|
| Logistic Regression  | 85%      | 0.873     |
| KNN (k=9)            | 82.5%    | 0.855     |
| Decision Tree        | 85%      | 0.885     |
| Random Forest        | 86.7%    | 0.905     |
| SVM                  | (Not shown) | (Not shown) |
| Gradient Boosting    | (Not shown) | (Not shown) |

*See page 14–17 for full evaluation details.*

---

## Future Work

- Integrate fairness auditing frameworks
- Expand feature engineering and hyperparameter tuning
- Deploy as a web API or dashboard for real-world testing

---

## Repository Structure

├── CreditApprovalDataset.ipynb
├── README.md
├── figures/ # (optional) ROC curves, plots
├── data/ # original and cleaned datasets
└── models/ # saved model files

---

## Authors

- **Nathan Nguyen** – [GitHub](https://github.com/nathannguyen33)  
- **Charlie Nguyen** – Collaborator

---