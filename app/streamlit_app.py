import streamlit as st
import joblib
import pandas as pd

# Page settings
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")

# Load model and scaler
model = joblib.load("../model/churn_model.pkl")
scaler = joblib.load("../model/scaler.pkl")

st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a telecom customer will **leave the service (churn)**.")

st.divider()

# ---------- Feature Columns ----------
feature_columns = [
'SeniorCitizen','tenure','MonthlyCharges','TotalCharges','gender_Male',
'Partner_Yes','Dependents_Yes','PhoneService_Yes',
'MultipleLines_No phone service','MultipleLines_Yes',
'InternetService_Fiber optic','InternetService_No',
'OnlineSecurity_No internet service','OnlineSecurity_Yes',
'OnlineBackup_No internet service','OnlineBackup_Yes',
'DeviceProtection_No internet service','DeviceProtection_Yes',
'TechSupport_No internet service','TechSupport_Yes',
'StreamingTV_No internet service','StreamingTV_Yes',
'StreamingMovies_No internet service','StreamingMovies_Yes',
'Contract_One year','Contract_Two year',
'PaperlessBilling_Yes',
'PaymentMethod_Credit card (automatic)',
'PaymentMethod_Electronic check',
'PaymentMethod_Mailed check'
]

# ---------- Layout ----------
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Customer Info")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

    tenure = st.slider("Tenure (months)", 0, 72)

with col2:
    st.subheader("📡 Service Details")

    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    payment_method = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Credit card (automatic)",
            "Bank transfer (automatic)"
        ]
    )

monthly_charges = st.number_input("Monthly Charges", value=70.0)

st.divider()

# ---------- Prediction ----------
if st.button("🔍 Predict Churn"):

    input_data = pd.DataFrame([[0]*len(feature_columns)], columns=feature_columns)

    input_data["tenure"] = tenure
    input_data["MonthlyCharges"] = monthly_charges
    input_data["TotalCharges"] = tenure * monthly_charges

    if senior == "Yes":
        input_data["SeniorCitizen"] = 1

    if gender == "Male":
        input_data["gender_Male"] = 1

    if partner == "Yes":
        input_data["Partner_Yes"] = 1

    if dependents == "Yes":
        input_data["Dependents_Yes"] = 1

    if phone_service == "Yes":
        input_data["PhoneService_Yes"] = 1

    if multiple_lines == "Yes":
        input_data["MultipleLines_Yes"] = 1
    elif multiple_lines == "No phone service":
        input_data["MultipleLines_No phone service"] = 1

    if internet_service == "Fiber optic":
        input_data["InternetService_Fiber optic"] = 1
    elif internet_service == "No":
        input_data["InternetService_No"] = 1

    if contract == "One year":
        input_data["Contract_One year"] = 1
    elif contract == "Two year":
        input_data["Contract_Two year"] = 1

    if paperless == "Yes":
        input_data["PaperlessBilling_Yes"] = 1

    if payment_method == "Electronic check":
        input_data["PaymentMethod_Electronic check"] = 1
    elif payment_method == "Mailed check":
        input_data["PaymentMethod_Mailed check"] = 1
    elif payment_method == "Credit card (automatic)":
        input_data["PaymentMethod_Credit card (automatic)"] = 1

    # Scale input
    scaled_input = scaler.transform(input_data)

    prediction = model.predict(scaled_input)
    probability = model.predict_proba(scaled_input)[0][1]

    st.divider()
    st.subheader("📊 Prediction Result")

    if prediction[0] == 1:
        st.error("⚠ Customer likely to churn")
    else:
        st.success("✅ Customer likely to stay")

    st.write(f"### Churn Probability: {probability:.2%}")

    # Risk meter
    st.progress(float(probability))