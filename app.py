import streamlit as st
import pandas as pd
import requests

API_URL = "http://127.0.0.1:8000/predict"  

st.title("Heart Disease Prediction App❤️")

# ===== USER INPUTS =====
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ASY", "ATA", "NAP"])
resting_bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
cholesterol = st.number_input("Cholesterol", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["ST"])   # only ST allowed
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Down", "Flat", "Up"])

# ===== PREDICT BUTTON =====
if st.button("Predict"):
   payload = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,    
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }
   response = requests.post(API_URL, json=payload)
   result=response.json()
   pred=result["Prediction"]
   pred_proba=result["Probability of Heart Disease"]
   st.write(f"**Disease Probability: {pred_proba*100:.2f}%**")

   if pred == 1:
        st.error("⚠️ HIGH RISK of Heart Disease")
   else:
        st.success("✅ LOW RISK of Heart Disease")
