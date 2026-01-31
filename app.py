import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
try:
    with open("stroke_model.pkl", "rb") as model_file:
        model = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model file not found! Train and save your model as 'stroke_model.pkl'")
    st.stop()

# Load the saved scaler to ensure consistent scaling
try:
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Scaler file not found! Train and save your scaler as 'scaler.pkl'")
    st.stop()

st.title("Stroke Risk Prediction")
st.write("Enter your details below to predict your stroke risk.")

# User inputs
gender = st.selectbox("Gender", ["Male", "Female", "Other"])
age = st.number_input("Age", min_value=1, max_value=120, value=30)
hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
ever_married = st.selectbox("Ever Married", ["No", "Yes"])
work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"])
residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
avg_glucose_level = st.number_input("Average Glucose Level", min_value=50.0, max_value=300.0, value=100.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0)
smoking_status = st.selectbox("Smoking Status", ["Never smoked", "Formerly smoked", "Smokes", "Unknown"])

# Label encoding for categorical variables
label_encoders = {
    "gender": LabelEncoder().fit(["Male", "Female", "Other"]),
    "ever_married": LabelEncoder().fit(["No", "Yes"]),
    "work_type": LabelEncoder().fit(["Private", "Self-employed", "Govt_job", "Children", "Never_worked"]),
    "residence_type": LabelEncoder().fit(["Urban", "Rural"]),
    "smoking_status": LabelEncoder().fit(["Never smoked", "Formerly smoked", "Smokes", "Unknown"])
}

# Preprocess user input
def preprocess_input():
    input_data = np.array([
        label_encoders["gender"].transform([gender])[0],
        age,
        1 if hypertension == "Yes" else 0,
        1 if heart_disease == "Yes" else 0,
        label_encoders["ever_married"].transform([ever_married])[0],
        label_encoders["work_type"].transform([work_type])[0],
        label_encoders["residence_type"].transform([residence_type])[0],
        avg_glucose_level,
        bmi,
        label_encoders["smoking_status"].transform([smoking_status])[0]
    ]).reshape(1, -1)

    input_data = scaler.transform(input_data)  # Use the preloaded scaler
    return input_data

# Predict stroke risk
if st.button("Predict Stroke Risk"):
    user_data = preprocess_input()
    proba = model.predict_proba(user_data)[0][1]  # Probability of stroke
    threshold = 0.3  # Adjusted threshold for better risk assessment

    st.write("### Prediction:")
    if proba > threshold:
        st.error(f"High Risk of Stroke! ({proba:.2%} probability)")
    else:
        st.success(f"Low Risk of Stroke ({proba:.2%} probability). Maintain a healthy lifestyle!")
