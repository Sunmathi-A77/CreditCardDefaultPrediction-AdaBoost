import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load trained AdaBoost model
# -------------------------------
with open('credit_default_model.pkl', 'rb') as f:
    model = pickle.load(f)

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(page_title="💳 Credit Card Default Prediction", page_icon="💳", layout="wide")

# -------------------------------
# Custom background color & emoji style
# -------------------------------
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;  /* light blue background */
    }
    .stApp {
        background-color: #f0f8ff;
    }
    h1 {
        color: #1a237e;  /* dark blue title */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Title and description
# -------------------------------
st.title("💳 Credit Card Default Prediction 💳")
st.write("Predict whether a customer will default on their credit card payment next month. 📊💰")

# -------------------------------
# Preprocessing function
# -------------------------------
def preprocess_input(df):
    num_cols = [
        'LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    for col in num_cols:
        lower = df[col].quantile(0.01)
        upper = df[col].quantile(0.99)
        df[col] = np.clip(df[col], lower, upper)

    skewed_cols = [
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6'
    ]
    for col in skewed_cols:
        df[col] = df[col].clip(lower=0)
        df[col] = np.log1p(df[col])
    return df

# -------------------------------
# Input fields with columns
# -------------------------------
st.header("📋 Enter Customer Information")
col1, col2, col3, col4 = st.columns(4)

with col1:
    LIMIT_BAL = st.number_input("💳 Credit Limit (LIMIT_BAL)", min_value=0, value=20000)
    SEX = st.selectbox("👤 Gender (1 = Male, 2 = Female)", [1, 2])
    EDUCATION = st.selectbox("🎓 Education (1=Grad School, 2=University, 3=High School, 4=Others)", [1, 2, 3, 4])
    MARRIAGE = st.selectbox("💍 Marital Status (1=Married, 2=Single, 3=Others)", [1, 2, 3])
    AGE = st.number_input("🎂 Age", min_value=18, max_value=100, value=35)

with col2:
    PAY_0 = st.number_input("📅 PAY_0 (Sept)", min_value=-2, max_value=8, value=0)
    PAY_2 = st.number_input("📅 PAY_2 (Aug)", min_value=-2, max_value=8, value=0)
    PAY_3 = st.number_input("📅 PAY_3 (July)", min_value=-2, max_value=8, value=0)
    PAY_4 = st.number_input("📅 PAY_4 (June)", min_value=-2, max_value=8, value=0)
    PAY_5 = st.number_input("📅 PAY_5 (May)", min_value=-2, max_value=8, value=0)
    PAY_6 = st.number_input("📅 PAY_6 (April)", min_value=-2, max_value=8, value=0)

with col3:
    BILL_AMT1 = st.number_input("💰 Bill Amount 1", min_value=0, value=5000)
    BILL_AMT2 = st.number_input("💰 Bill Amount 2", min_value=0, value=5000)
    BILL_AMT3 = st.number_input("💰 Bill Amount 3", min_value=0, value=5000)
    BILL_AMT4 = st.number_input("💰 Bill Amount 4", min_value=0, value=5000)
    BILL_AMT5 = st.number_input("💰 Bill Amount 5", min_value=0, value=5000)
    BILL_AMT6 = st.number_input("💰 Bill Amount 6", min_value=0, value=5000)

with col4:
    PAY_AMT1 = st.number_input("💵 Payment Amount 1", min_value=0, value=2000)
    PAY_AMT2 = st.number_input("💵 Payment Amount 2", min_value=0, value=2000)
    PAY_AMT3 = st.number_input("💵 Payment Amount 3", min_value=0, value=2000)
    PAY_AMT4 = st.number_input("💵 Payment Amount 4", min_value=0, value=2000)
    PAY_AMT5 = st.number_input("💵 Payment Amount 5", min_value=0, value=2000)
    PAY_AMT6 = st.number_input("💵 Payment Amount 6", min_value=0, value=2000)

# -------------------------------
# Create DataFrame
# -------------------------------
input_data = pd.DataFrame({
    'LIMIT_BAL': [LIMIT_BAL], 'SEX': [SEX], 'EDUCATION': [EDUCATION], 'MARRIAGE': [MARRIAGE], 'AGE': [AGE],
    'PAY_0': [PAY_0], 'PAY_2': [PAY_2], 'PAY_3': [PAY_3], 'PAY_4': [PAY_4], 'PAY_5': [PAY_5], 'PAY_6': [PAY_6],
    'BILL_AMT1': [BILL_AMT1], 'BILL_AMT2': [BILL_AMT2], 'BILL_AMT3': [BILL_AMT3],
    'BILL_AMT4': [BILL_AMT4], 'BILL_AMT5': [BILL_AMT5], 'BILL_AMT6': [BILL_AMT6],
    'PAY_AMT1': [PAY_AMT1], 'PAY_AMT2': [PAY_AMT2], 'PAY_AMT3': [PAY_AMT3],
    'PAY_AMT4': [PAY_AMT4], 'PAY_AMT5': [PAY_AMT5], 'PAY_AMT6': [PAY_AMT6]
})

# -------------------------------
# Preprocess input
# -------------------------------
processed_data = preprocess_input(input_data)

# -------------------------------
# Predict
# -------------------------------
if st.button("🔍 Predict Default"):
    prediction = model.predict(processed_data)[0]
    probability = model.predict_proba(processed_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ The customer is **likely to default** next month. (Risk Probability: {probability:.2f})")
    else:
        st.success(f"✅ The customer is **not likely to default** next month. (Risk Probability: {probability:.2f})")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("💳 Model: AdaBoost (DecisionTree Base Estimator) | Created by Sunmathi")
