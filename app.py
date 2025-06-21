import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

def show_model_metrics(model_name, y_test, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)

    st.subheader(f"Performance Metrics for {model_name}")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    st.pyplot(fig)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format("{:.2f}"))

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Receiver Operating Characteristic')
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

    st.markdown(f"**Current threshold:** {threshold:.2f}")

model_dict = {
    "Logistic Regression": joblib.load("models/log_reg.pkl"),
    "KNN (k=13)": joblib.load("models/knn.pkl"),
    "Decision Tree": joblib.load("models/dtree.pkl"),
    "SVM": joblib.load("models/svc.pkl"),
    "XGBoost": joblib.load("models/xgboost.pkl")
}

st.title("Credit Card Default Prediction")
st.write("Upload your processed `.xls` file.")

uploaded_file = st.file_uploader("Upload input file (.xls)", type=["xls"])

if uploaded_file is not None:
    try:
        input_data = pd.read_excel(uploaded_file, sheet_name="Data")
        input_data = input_data.rename(columns={'default payment next month': 'def_pay'})
        input_data['EDUCATION'] = input_data['EDUCATION'].replace([0, 6], 5)
        input_data['MARRIAGE'] = input_data['MARRIAGE'].replace(0, 3)
        input_data = input_data.rename(columns={'PAY_0': 'PAY_1'})
        input_data['SEX_female'] = input_data['SEX'].apply(lambda x: 1 if x == 2 else 0)
        input_data = input_data.drop('SEX', axis=1)
        input_data['MARRIAGE_married'] = input_data['MARRIAGE'].apply(lambda x: 1 if x == 1 else 0)
        input_data['MARRIAGE_single'] = input_data['MARRIAGE'].apply(lambda x: 1 if x == 2 else 0)
        input_data = input_data.drop('MARRIAGE', axis=1)
        input_data['EDUCATION_graduate_school'] = input_data['EDUCATION'].apply(lambda x: 1 if x == 1 else 0)
        input_data['EDUCATION_university'] = input_data['EDUCATION'].apply(lambda x: 1 if x == 2 else 0)
        input_data['EDUCATION_high_school'] = input_data['EDUCATION'].apply(lambda x: 1 if x == 3 else 0)
        input_data['EDUCATION_others'] = input_data['EDUCATION'].apply(lambda x: 1 if x == 4 else 0)
        input_data = input_data.drop('EDUCATION', axis=1)

        Y = input_data['def_pay']
        X_input = input_data.drop('def_pay', axis=1)
        if 'ID' in X_input.columns:
            X_input = X_input.drop('ID', axis=1)

        X_train, X_test, y_train, y_test = train_test_split(X_input, Y, test_size=0.3, random_state=123)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model_name = st.selectbox("Select model to predict", list(model_dict.keys()), key="model_selector")
        model = model_dict[model_name]

        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]

        result_df = pd.DataFrame({
            'True Label': y_test.values,
            'Prediction': preds,
            'Probability (Default)': probs
        })

        st.subheader("Prediction Results")
        st.write(result_df)

        st.success("✅ Completed Prediction")

        if st.checkbox("Evaluate model performance on uploaded data"):
            show_model_metrics(model_name, y_test.values, probs)

    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.info("👈 Please upload a valid `.xls` file")
