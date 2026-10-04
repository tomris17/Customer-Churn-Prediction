# Customer Churn Prediction Project

[![Python](https://img.shields.io/badge/Python-3.13%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Model-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red.svg)](https://streamlit.io/)

This repository contains an end-to-end Machine Learning Classification project designed to predict whether a customer will leave a telecommunication or subscription service.

---

## Dataset Notice
*Note: The dataset used in this project (`customer_churn_prediction_dataset.csv`) can be acquired from Kaggle or standard telecom churn repositories.*

---

## Dataset Features
The dataset contains customer demographic details, service subscriptions, and account info:
* **customerID**: Unique customer identifier (dropped during preprocessing).
* **gender, SeniorCitizen, Partner, Dependents**: Demographic attributes.
* **tenure, PhoneService, InternetService, Contract**: Service-related attributes.
* **MonthlyCharges, TotalCharges**: Financial metrics.
* **Churn**: Target variable indicating service discontinuation.

---

## Project Workflow
1. **Data Preprocessing**: Dropping identifiers, encoding binary target labels (`Churn`), and applying one-hot encoding (`pd.get_dummies`).
2. **Model Benchmarking**: Automated evaluation (`algo_test`) of 8 classification models including BernoulliNB, Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, KNeighbors, AdaBoost, and MultinomialNB.
3. **Model Selection**: Training and selecting the optimal classifier (`GradientBoostingClassifier`).
4. **Evaluation**: Assessing performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
5. **Model Persistence**: Saving the trained model using `joblib` into `churn_prediction_model.pkl`.
6. **Web Application**: Interactive deployment interface built with Streamlit.

---

## Getting Started & Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
   cd customer-churn-prediction
