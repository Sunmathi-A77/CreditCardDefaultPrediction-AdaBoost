# 💳 Credit Card Default Prediction App

Predict whether a customer will default on their credit card payment next month using AdaBoost with a Decision Tree base estimator.

## 📝 Project Overview

Credit card default prediction is critical for financial institutions to assess the risk of lending and minimize losses. This project uses historical credit card client data to train a predictive model that can classify whether a customer is likely to default next month.

The project involves data preprocessing, handling outliers, reducing skewness, feature engineering, model training with hyperparameter tuning, and deployment via Streamlit for a user-friendly interface.

## 🔗 App Deployment Link

Credit Card Default Prediction App - (https://creditcarddefaultprediction-adaboost.streamlit.app/)

## 🛠 Dataset

Default of Credit Card Clients Dataset (Link - https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?utm_source=chatgpt.com)

Target variable: default

0 → No default

1 → Default

### Features include:

LIMIT_BAL, AGE, SEX, EDUCATION, MARRIAGE

Previous payment status: PAY_0 to PAY_6

Bill amounts: BILL_AMT1 to BILL_AMT6

Payment amounts: PAY_AMT1 to PAY_AMT6

## 🚀 Setup & Run Locally

1. Clone the repository

   git clone <your_repo_link>

   cd CreditCardDefaultPrediction-AdaBoost

2. Create and activate virtual environment

   python -m venv myvenv

   myvenv\Scripts\activate    # Windows

   source myvenv/bin/activate    # Mac/Linux

3. Install dependencies

   pip install -r requirements.txt

4. Run Streamlit App
   
   streamlit run app.py

## 🛠 Steps Performed

### 1. Data Preprocessing

Loaded the dataset and checked for missing values.

Clipped numerical columns (1st to 99th percentile) to handle outliers.

Log-transformed highly skewed columns (PAY_AMT1-6, BILL_AMT1-6) to reduce skewness.

### 2. Feature Selection

Target column: default (0 = No default, 1 = Default)

Features: LIMIT_BAL, AGE, SEX, EDUCATION, MARRIAGE, previous payment statuses, bill amounts, payment amounts.

### 3. Train-Test Split

Split dataset into train (80%) and test (20%) using stratification to maintain class distribution.

### 4. Model Training

Base estimator: Decision Tree (max_depth=3)

Model: AdaBoost Classifier

Hyperparameter tuning using GridSearchCV to find optimal:

n_estimators

learning_rate

### 5. Evaluation

Metrics calculated: Accuracy, Confusion Matrix, Classification Report, ROC-AUC

Visualized ROC Curve to evaluate model performance.

Selected best model based on ROC-AUC and accuracy.

### 6. Model Saving

Saved the trained model using pickle as credit_default_model.pkl.

### 7. Deployment

Built a Streamlit web app with:

User-friendly input forms for customer data

Real-time default prediction

Risk probability display

Color-coded messages (green = safe, red = risk)

Added emojis and background colors for UI

## 🧠 Model

Algorithm: AdaBoost Classifier

Base Estimator: Decision Tree (max_depth=3)

Hyperparameters:

n_estimators = 500

learning_rate = 0.1

## 📊 Model Performance

| Metric                | Value |
| --------------------- | ----- |
| Accuracy              | 0.817 |
| ROC-AUC               | 0.779 |
| Precision (Default=1) | 0.66  |
| Recall (Default=1)    | 0.35  |
| F1-Score (Default=1)  | 0.46  |

## 🗂 File Structure

CreditCardDefaultPrediction-AdaBoost/

│

├─ app.py                   # Streamlit application

├─ credit_default_model.pkl # Trained AdaBoost model

├─ notebook.ipynb           # Jupyter notebook for model training & evaluation

├─ requirements.txt         # Python dependencies

├─ README.md                # Project documentation

## 🎨 UI Features

Modern UI with columns for input fields

Color-coded messages: green (safe), red (risk)

Background color & emojis to improve UX

## 💻 App Interface

Credit Limit, Age, Gender, Education, Marital Status

Previous Payment Status: PAY_0 → PAY_6

Bill Amounts: BILL_AMT1 → BILL_AMT6

Payment Amounts: PAY_AMT1 → PAY_AMT6

Predict Button: Shows result

✅ Customer is not likely to default

⚠️ Customer is likely to default

Displays risk probability (0 to 1)

## ⚙️ Technologies & Libraries Used

Python – Programming language

Jupyter Notebook – Data analysis and model training

Streamlit – Web app deployment

### Libraries:

pandas, numpy – Data manipulation

matplotlib, seaborn – Visualization

scikit-learn – Machine Learning, AdaBoost, Decision Trees, GridSearchCV

pickle – Saving and loading the trained model

## 📌 Notes

Model is trained with historical credit card client data.

Prediction probability helps banks assess risk before issuing credit.

Make sure credit_default_model.pkl exists in the same folder as app.py.

Preprocessing steps in the app must match those used during training.
