import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np

# --- Function to Display Model Metrics ---
def show_model_metrics(model_name, y_true, y_pred, y_proba):
    """Displays classification metrics and visualizations."""
    st.write("---")
    st.subheader(f"Performance Metrics for: {model_name}")

    # Confusion Matrix
    st.write("#### Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    # Classification Report
    st.write("#### Classification Report")
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    # Apply styling for better readability
    st.dataframe(report_df.style.format({
        "precision": "{:.2f}",
        "recall": "{:.2f}",
        "f1-score": "{:.2f}",
        "support": "{:.0f}"
    }))

    # ROC Curve
    st.write("#### ROC Curve")
    fpr, tpr, _ = roc_curve(y_true, y_proba)
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

# --- Main App ---

# Load pre-trained models and the scaler
# It's good practice to wrap this in a try-except block
try:
    model_dict = {
        "Logistic Regression": joblib.load("models/log_reg.pkl"),
        "KNN (k=13)": joblib.load("models/knn.pkl"),
        "Decision Tree": joblib.load("models/dtree.pkl"),
        "SVM": joblib.load("models/svc.pkl"),
        "XGBoost": joblib.load("models/xgboost.pkl")
    }
    # IMPORTANT: The scaler must be loaded here to apply the SAME transformation
    # that the models were trained on.
    scaler = joblib.load("models/scaler.pkl")
except FileNotFoundError:
    st.error("❌ Critical Error: Model or scaler files not found in the 'models/' directory. Please ensure they exist.")
    st.stop() # Stop the app if essential files are missing

# App Title
st.title("💳 Credit Card Default Prediction")
st.write("Upload a preprocessed `.xls` file to predict customer defaults.")

# File uploader
uploaded_file = st.file_uploader("Upload input file", type=["xls", "xlsx"])

# Only proceed if a file is uploaded
if uploaded_file is not None:
    try:
        input_data = pd.read_excel(uploaded_file, sheet_name="Data")
        st.write("#### Uploaded Data Preview:")
        st.dataframe(input_data.head())

        # --- Preprocessing Pipeline ---
        # This should be the same preprocessing as your training script
        processed_data = input_data.copy()
        processed_data = processed_data.rename(columns={'default payment next month': 'def_pay', 'PAY_0': 'PAY_1'})
        processed_data['EDUCATION'] = processed_data['EDUCATION'].replace([0, 5, 6], 4) # Consolidate 'other' categories
        processed_data['MARRIAGE'] = processed_data['MARRIAGE'].replace(0, 3)

        # One-hot encode categorical features
        # This ensures the column order and names match the training data
        processed_data['SEX_female'] = (processed_data['SEX'] == 2).astype(int)
        processed_data = processed_data.drop('SEX', axis=1)

        for val, name in {1: 'married', 2: 'single', 3: 'others'}.items():
            processed_data[f'MARRIAGE_{name}'] = (processed_data['MARRIAGE'] == val).astype(int)
        processed_data = processed_data.drop('MARRIAGE', axis=1)

        for val, name in {1: 'graduate_school', 2: 'university', 3: 'high_school', 4: 'others'}.items():
             processed_data[f'EDUCATION_{name}'] = (processed_data['EDUCATION'] == val).astype(int)
        processed_data = processed_data.drop('EDUCATION', axis=1)

        # Separate features (X) and target (Y)
        # We assume the uploaded file contains the target column for evaluation purposes
        if 'def_pay' not in processed_data.columns:
            st.error("Target column 'default payment next month' (renamed to 'def_pay') not found in uploaded file.")
            st.stop()
            
        Y_input = processed_data['def_pay']
        X_input = processed_data.drop(['def_pay', 'ID'], axis=1)

        # Ensure column order matches the model's training data
        # This is a critical step to prevent errors
        expected_columns = scaler.get_feature_names_out()
        X_input = X_input[expected_columns]

        # --- Scaling and Prediction ---
        # Use the LOADED scaler to transform the new data
        X_input_scaled = scaler.transform(X_input)

        # Model selection dropdown
        model_name = st.selectbox(
            "Select a model for prediction:",
            list(model_dict.keys()),
            key="model_selector"  # A unique key prevents the error
        )
        model = model_dict[model_name]

        # Make predictions
        predictions = model.predict(X_input_scaled)
        probabilities = model.predict_proba(X_input_scaled)[:, 1]

        # --- Display Results ---
        st.write("---")
        st.subheader("Prediction Results")
        result_df = pd.DataFrame({
            'Original_ID': input_data['ID'],
            'Probability_of_Default': probabilities,
            'Prediction': predictions
        })
        result_df['Prediction'] = result_df['Prediction'].apply(lambda x: "Default" if x == 1 else "No Default")
        st.dataframe(result_df)
        st.success("✅ Prediction complete.")
        
        # --- Optional: Evaluate Model Performance ---
        if st.checkbox("Evaluate model performance on the uploaded data"):
            show_model_metrics(model_name, Y_input, predictions, probabilities)

    except Exception as e:
        st.error(f"❌ An error occurred during processing: {e}")
else:
    st.info("ℹ️ Please upload a valid `.xls` or `.xlsx` file to begin.")
