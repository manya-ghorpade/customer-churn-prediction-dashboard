# 📊 Customer Churn Prediction Dashboard

A Machine Learning project that predicts whether a telecom customer is likely to churn using customer demographics, service usage, and account information.

The project includes a trained ML model and an interactive dashboard built with Streamlit.

---

## 🚀 Live Demo

Streamlit App:  
https://customer-churn-prediction-dashboard-qrtsogc4kdqxn8maks6hbb.streamlit.app/

---

## 📌 Project Overview

Customer churn is a major challenge for telecom companies because losing customers directly impacts revenue.

This project builds a **machine learning model to predict churn risk** and provides an **interactive dashboard** for analyzing customer behavior and churn probability.

The application allows users to input customer information and instantly see the churn prediction.

---

## 📊 Dataset

Dataset used: **Telco Customer Churn Dataset**

Source:  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

The dataset contains information about:

- Customer demographics
- Account information
- Services subscribed
- Billing details
- Churn status

---

## 🧠 Machine Learning Workflow

1. Data Cleaning and Preprocessing  
2. Feature Encoding  
3. Feature Scaling using StandardScaler  
4. Model Training  
5. Model Evaluation  
6. Deployment using Streamlit  

---

## 🤖 Models Used

- Logistic Regression
- Random Forest
- XGBoost

Final model selected: **Logistic Regression**

Model Performance:

- Accuracy: ~82%
- F1 Score: ~0.64
- ROC-AUC: ~0.75

---

## 🖥 Dashboard Features

The Streamlit dashboard provides:

- Customer input panel
- Churn probability prediction
- Risk level indicator
- Customer analytics charts
- Interactive UI

Users can adjust customer details such as:

- Tenure
- Monthly Charges
- Contract Type
- Payment Method
- Services used

and instantly view churn predictions.

---

## 🗂 Project Structure

```
customer-churn-prediction-dashboard
│
├── app
│   └── streamlit_app.py
│
├── model
│   ├── churn_model.pkl
│   └── scaler.pkl
│
├── notebook
│   └── churn_prediction.ipynb
│
├── .streamlit
│   └── config.toml
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

Clone the repository:

```
git clone https://github.com/manya-ghorpade/customer-churn-prediction-dashboard.git
```

Navigate to the project directory:

```
cd customer-churn-prediction-dashboard
```

Create virtual environment:

```
python -m venv venv
```

Activate environment:

Windows:

```
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the Streamlit app:

```
cd app
streamlit run streamlit_app.py
```

## 🔮 Future Improvements

- Hyperparameter tuning
- SHAP model explainability
- Advanced churn analytics
- Customer segmentation
- Deployment using Docker

---

## 👨‍💻 Author

**Manya Ghorpade**

BTech AI & ML Student

GitHub:  
https://github.com/manya-ghorpade
