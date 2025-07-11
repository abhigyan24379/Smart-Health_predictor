import streamlit as st
import pandas as pd 
import numpy as np
import joblib
import os
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "../models/rf_model.pkl")
model = joblib.load(model_path)
st.set_page_config(page_title="Smart Health Prediction", layout="centered")

st.title("Smart Health Predictor")
st.write("Enter the patient details below: to predict the diabetes")

#inputs
pregnancies = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
blood_pressure = st.number_input("Blood Pressure", min_value=0)
skin_thickness = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("Age", min_value=0)

if st.button("Predict"):
    # creating input dataframe 
    input_data = pd.DataFrame(
        {
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],  
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [diabetes_pedigree],
            'Age': [age]
        }
    )
    # Predict 
    prediction = model.predict(input_data)[0]
    result = "Not Diabetic" if prediction == 0 else "Diabetic"
    
    st.subheader("Prediction Result")
    st.success(result)
    
    










