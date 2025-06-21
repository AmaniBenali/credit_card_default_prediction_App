import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

# --- Page Configuration ---
st.set_page_config(
    page_title="Credit Default Prediction",
    page_icon="💳",
    layout="wide"
)

# --- Caching Functions for Performance ---

@st.cache_data
def load_data(uploaded_file):
    """Loads data from an uploaded Excel file."""
    if uploaded_file is not None:
        try:
            # To read from file-like object
            data = pd.read_excel(uploaded_file, sheet_name='Data', header=1)
            return data
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None

@st.cache_data
def preprocess(df):
    """Applies all the preprocessing steps to the dataframe."""
    df_processed = df.copy()
    
    # Rename columns
    df_processed = df_processed.rename(columns={'default payment next month': 'def_pay', 'PAY_0': 'PAY_1'})
    
    # Clean EDUCATION: Combine 0, 5, and 6 into a single 'unknown' category (represented by 5)
    df_processed['EDUCATION'] = df_processed['EDUCATION'].replace([0, 6], 5)

    # One-hot encode categorical features
    df_processed['SEX_female'] = df_processed['SEX'].apply(lambda x: 1 if x == 2 else 0)
    df_processed = df_processed.drop('SEX', axis=1)

    df_processed['MARRIAGE_married'] = df_processed['MARRIAGE'].apply(lambda x: 1 if x == 1 else 0)
    df_processed['MARRIAGE_single'] = df_processed['MARRIAGE'].apply(lambda x: 1 if x == 2 else 0)
    df_processed = df_processed.drop('MARRIAGE', axis=1)

    # One-hot encode EDUCATION
    edu_dummies = pd.get_dummies(df_processed['EDUCATION'], prefix='EDUCATION')
    # Rename for clarity
    edu_dummies = edu_dummies.rename(columns={
        'EDUCATION_1': 'EDUCATION_graduate_school',
        'EDUCATION_2': 'EDUCATION_university',
        'EDUCATION_3': 'EDUCATION_high_school',
        'EDUCATION_4': 'EDUCATION_others',
        'EDUCATION_5': 'EDUCATION_unknown'
    })
    df_processed = pd.concat([df_processed, edu_dummies], axis=1)
    df_processed = df_processed.drop('EDUCATION', axis=1)
    
    return df_processed

@st.cache_resource
def balance_data(X, y):
    """Balances the data using SMOTE."""
    sm = SMOTE(random_state=123)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

# --- Main App Logic ---

st.title("💳 Credit Card Default Prediction Dashboard")
st.markdown("""
This interactive dashboard allows you to explore the credit card default dataset, visualize model performance, and make real-time predictions.
Upload your data or use the sample to get started.
""")

# --- Sidebar for Inputs ---
with st.sidebar:
    st.header("⚙️ Controls & Inputs")

    # File Uploader
    uploaded_file = st.file_uploader("Choose an Excel file", type="xls")
    
    if uploaded_file is None:
        st.info("A sample dataset is used by default. Upload your own `.xls` file to analyze.")

    # Model Selection
    st.markdown("### เลือกแบบจำลอง")
    model_choice = st.selectbox(
        "Choose a model to evaluate:",
        ("Logistic Regression", "K-Nearest Neighbors", "Decision Tree", "Support Vector Machine (SVM)", "XGBoost")
    )
    
    # Conditional slider for KNN
    k_value = 13 # Default value from your code
    if model_choice == 'K-Nearest Neighbors':
        k_value = st.slider("Select K value for KNN", min_value=1, max_value=20, value=13, step=1)

    st.markdown("---")
    st.header("👤 Predict for a New Client")
    st.markdown("Adjust the values below to predict if a new client will default.")
    
    # Input fields for prediction
    limit_bal = st.number_input('Credit Limit (LIMIT_BAL)', min_value=10000, max_value=1000000, value=50000, step=1000)
    age = st.slider('Age', 18, 100, 35)
    sex_female = st.selectbox('Gender (SEX)', options=[0, 1], format_func=lambda x: 'Female' if x == 1 else 'Male')
    
    education_map = {'Graduate School': 1, 'University': 2, 'High School': 3, 'Others': 4, 'Unknown': 5}
    education_val = st.selectbox('Education Level', options=list(education_map.keys()))
    
    marriage_map = {'Married': 1, 'Single': 2, 'Others': 0}
    marriage_val = st.selectbox('Marriage Status', options=list(marriage_map.keys()))

    # Advanced Features Expander
    with st.expander("Advanced Features (Payment Status & Bill Amounts)"):
        pay_cols = st.columns(6)
        pay_status = []
        for i, col in enumerate(pay_cols):
             pay_status.append(col.number_input(f'PAY_{i+1}', min_value=-2, max_value=8, value=0, key=f'pay_{i}'))

        bill_cols = st.columns(6)
        bill_amts = []
        for i, col in enumerate(bill_cols):
            bill_amts.append(col.number_input(f'BILL_AMT{i+1}', min_value=-150000, max_value=1000000, value=20000, key=f'bill_{i}'))
        
        pay_amt_cols = st.columns(6)
        pay_amts = []
        for i, col in enumerate(pay_amt_cols):
            pay_amts.append(col.number_input(f'PAY_AMT{i+1}', min_value=0, max_value=1000000, value=1000, key=f'pay_amt_{i}'))


# --- Data Loading and Processing ---
if uploaded_file:
    data = load_data(uploaded_file)
else:
    # As a fallback, create a dummy file object for the default file path for caching to work
    try:
        with open('/content/default of credit card clients.xls', 'rb') as f:
            data = load_data(f)
    except FileNotFoundError:
        st.error("Default file not found. Please upload a dataset to proceed.")
        st.stop()

if data is not None:
    data_processed = preprocess(data)
    
    X = data_processed.drop(['ID', 'def_pay'], axis=1)
    y = data_processed['def_pay']
    
    # Ensure column order is consistent for prediction
    X_columns = X.columns.tolist()

    X_res, y_res = balance_data(X, y)
    
    # Train-test split
    X_res_train, X_res_test, y_res_train, y_res_test = train_test_split(X_res, y_res, test_size=0.3, random_state=123)
    
    # Scale data
    scaler = StandardScaler()
    X_res_train_stand = scaler.fit_transform(X_res_train)
    X_res_test_stand = scaler.transform(X_res_test)

    # --- Main Panel for Display ---
    
    # 1. Data Overview
    st.header("📊 Data Overview")
    with st.expander("Click to see the raw and processed data"):
        st.subheader("Raw Data Sample")
        st.dataframe(data.head())
        st.subheader("Processed Data Sample (with one-hot encoding)")
        st.dataframe(data_processed.head())
        
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Class Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='def_pay', data=data, ax=ax)
        ax.set_xticklabels(['No Default', 'Default'])
        ax.set_title('Default Payment - Before SMOTE')
        st.pyplot(fig)
    with col2:
        st.subheader("Balanced Class Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=y_res, ax=ax)
        ax.set_xticklabels(['No Default', 'Default'])
        ax.set_title('Default Payment - After SMOTE')
        st.pyplot(fig)

    st.markdown("---")
    
    # 2. Model Evaluation
    st.header(f"📈 Model Performance: {model_choice}")
    
    # Train the selected model
    if model_choice == 'Logistic Regression':
        model = LogisticRegression(solver='liblinear', random_state=123)
    elif model_choice == 'K-Nearest Neighbors':
        model = KNeighborsClassifier(n_neighbors=k_value)
    elif model_choice == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=123)
    elif model_choice == 'Support Vector Machine (SVM)':
        model = SVC(probability=True, random_state=123)
    elif model_choice == 'XGBoost':
        model = xgb.XGBClassifier(random_state=123, use_label_encoder=False, eval_metric='logloss')

    model.fit(X_res_train_stand, y_res_train)
    y_pred = model.predict(X_res_test_stand)
    y_pred_proba = model.predict_proba(X_res_test_stand)[:, 1]
    accuracy = accuracy_score(y_res_test, y_pred)

    # Display Metrics
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_res_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['No Default', 'Default'], yticklabels=['No Default', 'Default'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

    with m_col2:
        st.subheader("Performance Metrics")
        st.metric(label="Accuracy Score", value=f"{accuracy:.4f}")
        st.text("Classification Report:")
        report = classification_report(y_res_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    # ROC Curve
    st.subheader("Receiver Operating Characteristic (ROC) Curve")
    fpr, tpr, _ = roc_curve(y_res_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve for {model_choice}')
    ax.legend(loc="lower right")
    st.pyplot(fig)

    st.markdown("---")
    
    # 3. Prediction Section
    st.header("🔮 Make a Prediction")

    if st.button("Predict Client Default Status", type="primary"):
        # Create a dictionary for the new client's data
        client_data = {
            'LIMIT_BAL': limit_bal, 'AGE': age,
            'PAY_1': pay_status[0], 'PAY_2': pay_status[1], 'PAY_3': pay_status[2],
            'PAY_4': pay_status[3], 'PAY_5': pay_status[4], 'PAY_6': pay_status[5],
            'BILL_AMT1': bill_amts[0], 'BILL_AMT2': bill_amts[1], 'BILL_AMT3': bill_amts[2],
            'BILL_AMT4': bill_amts[3], 'BILL_AMT5': bill_amts[4], 'BILL_AMT6': bill_amts[5],
            'PAY_AMT1': pay_amts[0], 'PAY_AMT2': pay_amts[1], 'PAY_AMT3': pay_amts[2],
            'PAY_AMT4': pay_amts[3], 'PAY_AMT5': pay_amts[4], 'PAY_AMT6': pay_amts[5],
            'SEX_female': sex_female,
            'MARRIAGE_married': 1 if marriage_map[marriage_val] == 1 else 0,
            'MARRIAGE_single': 1 if marriage_map[marriage_val] == 2 else 0,
            'EDUCATION_graduate_school': 1 if education_map[education_val] == 1 else 0,
            'EDUCATION_university': 1 if education_map[education_val] == 2 else 0,
            'EDUCATION_high_school': 1 if education_map[education_val] == 3 else 0,
            'EDUCATION_others': 1 if education_map[education_val] == 4 else 0,
            'EDUCATION_unknown': 1 if education_map[education_val] == 5 else 0
        }
        
        # Create DataFrame and ensure column order
        client_df = pd.DataFrame([client_data])
        client_df = client_df[X_columns] # Important: enforce same column order as training
        
        # Scale the data using the FITTED scaler
        client_df_stand = scaler.transform(client_df)
        
        # Make prediction
        prediction = model.predict(client_df_stand)
        prediction_proba = model.predict_proba(client_df_stand)
        
        # Display result
        st.subheader("Prediction Result")
        if prediction[0] == 1:
            st.error(f"Prediction: **Default** (Probability: {prediction_proba[0][1]:.2f})")
        else:
            st.success(f"Prediction: **No Default** (Probability: {prediction_proba[0][0]:.2f})")

