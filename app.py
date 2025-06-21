import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np


from sklearn.preprocessing import StandardScaler
def show_model_metrics(model_name, y_test, y_proba, threshold=0.5):
    # Apply threshold to probability to get predictions
    y_pred = (y_proba >= threshold).astype(int)

    st.subheader(f"Performance Metrics for {model_name}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic')
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

    # Display current threshold
    st.markdown(f"**Current threshold:** {threshold:.2f}")
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
st.write("Upload your processed `.xls` file .")

# File uploader
uploaded_file = st.file_uploader("Upload input file (.xls)", type=["xls"])

if uploaded_file is not None:
    try:
        input_data = pd.read_excel(uploaded_file, sheet_name="Data")

        # Clean and preprocess just like training
        input_data = input_data.rename(columns={'default payment next month': 'def_pay', 'PAY_0': 'PAY_1'})
        input_data['EDUCATION'] = input_data['EDUCATION'].replace([0, 6], 5)
        input_data['MARRIAGE'] = input_data['MARRIAGE'].replace(0, 3)
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

# Ask user to upload actual labels for evaluation
if st.checkbox("Evaluate model performance on uploaded data"):
            y_true = input_data['def_pay'].values  # True labels from uploaded data
            show_model_metrics(model_name, y_true, probs)


