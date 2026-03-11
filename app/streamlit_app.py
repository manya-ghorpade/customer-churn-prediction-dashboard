import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

model = joblib.load("model/churn_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# ---------------- FEATURE LIST ----------------

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

# ---------------- HEADER ----------------

st.title("Customer Churn Prediction Dashboard")
st.caption("Machine Learning system that predicts telecom customer churn risk")

st.divider()

# ---------------- DASHBOARD METRICS ----------------

col1,col2,col3 = st.columns(3)

col1.metric("Customers","7043")
col2.metric("Churn Rate","26.5%")
col3.metric("Model Accuracy","82%")

st.divider()

# ---------------- SIDEBAR ----------------

st.sidebar.header("Customer Information")

gender = st.sidebar.selectbox("Gender",["Male","Female"])
senior = st.sidebar.selectbox("Senior Citizen",["Yes","No"])
partner = st.sidebar.selectbox("Partner",["Yes","No"])
dependents = st.sidebar.selectbox("Dependents",["Yes","No"])

st.sidebar.header("Account Details")

tenure = st.sidebar.slider("Tenure (months)",0,72,12)

monthly_charges = st.sidebar.number_input("Monthly Charges",value=70.0)

contract = st.sidebar.selectbox(
"Contract Type",
["Month-to-month","One year","Two year"]
)

paperless = st.sidebar.selectbox(
"Paperless Billing",
["Yes","No"]
)

payment_method = st.sidebar.selectbox(
"Payment Method",
[
"Electronic check",
"Mailed check",
"Credit card (automatic)",
"Bank transfer (automatic)"
]
)

st.sidebar.header("Services")

phone_service = st.sidebar.selectbox("Phone Service",["Yes","No"])

multiple_lines = st.sidebar.selectbox(
"Multiple Lines",
["No","Yes","No phone service"]
)

internet_service = st.sidebar.selectbox(
"Internet Service",
["DSL","Fiber optic","No"]
)

online_security = st.sidebar.selectbox(
"Online Security",
["No","Yes","No internet service"]
)

online_backup = st.sidebar.selectbox(
"Online Backup",
["No","Yes","No internet service"]
)

device_protection = st.sidebar.selectbox(
"Device Protection",
["No","Yes","No internet service"]
)

tech_support = st.sidebar.selectbox(
"Tech Support",
["No","Yes","No internet service"]
)

streaming_tv = st.sidebar.selectbox(
"Streaming TV",
["No","Yes","No internet service"]
)

streaming_movies = st.sidebar.selectbox(
"Streaming Movies",
["No","Yes","No internet service"]
)

predict_button = st.sidebar.button("Predict Churn")

# ---------------- CREATE INPUT VECTOR ----------------

input_data = pd.DataFrame([[0]*len(feature_columns)], columns=feature_columns)

input_data["tenure"] = tenure
input_data["MonthlyCharges"] = monthly_charges
input_data["TotalCharges"] = tenure * monthly_charges

if gender=="Male":
    input_data["gender_Male"]=1

if senior=="Yes":
    input_data["SeniorCitizen"]=1

if partner=="Yes":
    input_data["Partner_Yes"]=1

if dependents=="Yes":
    input_data["Dependents_Yes"]=1

if phone_service=="Yes":
    input_data["PhoneService_Yes"]=1

if multiple_lines=="Yes":
    input_data["MultipleLines_Yes"]=1
elif multiple_lines=="No phone service":
    input_data["MultipleLines_No phone service"]=1

if internet_service=="Fiber optic":
    input_data["InternetService_Fiber optic"]=1
elif internet_service=="No":
    input_data["InternetService_No"]=1

if contract=="One year":
    input_data["Contract_One year"]=1
elif contract=="Two year":
    input_data["Contract_Two year"]=1

if paperless=="Yes":
    input_data["PaperlessBilling_Yes"]=1

if payment_method=="Electronic check":
    input_data["PaymentMethod_Electronic check"]=1
elif payment_method=="Mailed check":
    input_data["PaymentMethod_Mailed check"]=1
elif payment_method=="Credit card (automatic)":
    input_data["PaymentMethod_Credit card (automatic)"]=1

# ---------------- MAIN DASHBOARD ----------------

left,right = st.columns([2,1])

# ---------------- ANALYTICS ----------------

with left:

    st.subheader("Customer Analytics")

    churn_counts=[5174,1869]

    fig=px.pie(
        values=churn_counts,
        names=["Stay","Churn"],
        hole=0.6,
        color_discrete_sequence=["#22c55e","#ef4444"]
    )

    st.plotly_chart(fig,use_container_width=True)

# ---------------- PREDICTION PANEL ----------------

with right:

    st.subheader("Prediction")

    if predict_button:

        scaled_input = scaler.transform(input_data)

        prediction = model.predict(scaled_input)
        prob = model.predict_proba(scaled_input)[0][1]

        st.metric("Churn Probability",f"{prob*100:.2f}%")

        if prob>0.6:
            st.error("High churn risk")
        else:
            st.success("Customer likely to stay")

        st.progress(float(prob))