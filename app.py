import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.preprocessing import StandardScaler

# Load models and scaler
model_dict = {
    "Logistic Regression": joblib.load("models/log_reg.pkl"),
    "KNN (k=13)": joblib.load("models/knn.pkl"),
    "Decision Tree": joblib.load("models/dtree.pkl"),
    "SVM": joblib.load("models/svc.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl")
}
scaler = joblib.load("models/scaler.pkl")

# Title
st.title("Credit Card Default Prediction")
st.write("Upload your processed `.xlsx` file (with same format used during training).")

# File uploader
uploaded_file = st.file_uploader("Upload input file (.xlsx)", type=["xlsx"])

if uploaded_file is not None:
    try:
        input_data = pd.read_excel(uploaded_file, sheet_name="Data")

        # Clean and preprocess just like training
        input_data = input_data.rename(columns={'default payment next month': 'def_pay', 'PAY_0': 'PAY_1'})
        input_data['EDUCATION'] = input_data['EDUCATION'].replace([0, 6], 5)
        input_data['SEX_female'] = input_data['SEX'].apply(lambda x: 1 if x == 2 else 0)
        input_data['MARRIAGE_married'] = input_data['MARRIAGE'].apply(lambda x: 1 if x == 1 else 0)
        input_data['MARRIAGE_single'] = input_data['MARRIAGE'].apply(lambda x: 1 if x == 2 else 0)
        input_data['EDUCATION_graduate_school'] = input_data['EDUCATION'].apply(lambda x: 1 if x == 1 else 0)
        input_data['EDUCATION_university'] = input_data['EDUCATION'].apply(lambda x: 1 if x == 2 else 0)
        input_data['EDUCATION_high_school'] = input_data['EDUCATION'].apply(lambda x: 1 if x == 3 else 0)
        input_data['EDUCATION_others'] = input_data['EDUCATION'].apply(lambda x: 1 if x == 4 else 0)

        input_data.drop(['SEX', 'MARRIAGE', 'EDUCATION'], axis=1, inplace=True)
        X_input = input_data.drop('def_pay', axis=1)
        X_input = X_input.drop('ID', axis=1)

        # Standardize
        X_scaled = scaler.transform(X_input)

        # Select model
        model_name = st.selectbox("Select model to predict", list(model_dict.keys()))
        model = model_dict[model_name]

        # Predict
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)[:, 1]

        # Show predictions
        result_df = pd.DataFrame({
            'Prediction': preds,
            'Probability (Default)': probs
        })
        st.subheader("Prediction Results")
        st.write(result_df)

        st.success("✅ Completed Prediction")
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👈 Please upload a valid `.xlsx` file")
