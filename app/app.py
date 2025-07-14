import streamlit as st
import pandas as pd 
import numpy as np
import joblib
import shap 
import matplotlib.pyplot as plt
import os

# page setup 
st.set_page_config(
    page_title="Smart Health predictor ", 
    layout="wide"
)

st.markdown("Smart Health Predictor")
st.markdown("This is a web application that predicts diabetes based on patient details using machine learning models.")
st.markdown("---")

# model selection 
model_choice = st.selectbox(
    "choose a  model :",
    ["Random Forest", "XGBoost", "Logistic Regression"]
    
)
model_map = {
    "Random Forest": "rf_model.pkl",
    "XGBoost": "xgb_model.pkl", 
    "Logistic Regression": "log_model.pkl"
}

base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, f"../models/{model_map[model_choice]}")
model = joblib.load(model_path)
scaler_path = os.path.join(base_path, "../models/scaler.pkl")
scaler = joblib.load(scaler_path)
# st.set_page_config(page_title="Smart Health Prediction", layout="centered")

# st.title("Smart Health Predictor")
# st.write("Enter the patient details below: to predict the diabetes")

col1 , col2 = st.columns(2) # two columns for layout


with col1:
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
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        # result = "Not Diabetic" if prediction == 0 else "Diabetic"
        prob = model.predict_proba(scaled_input)[0,1]
        
        with col2:
            st.subheader("Prediction Result")
            # st.success(result)
            if prediction == 1:
                st.error(f"Diabetic - Risk Probability : {prob:.2f}")
            else:
                st.success(f"Not Diabetic - Risk Probability : {prob:.2f}")
                
            with st.expander("View Shap Explanation"):
                explainer = shap.Explainer(model, input_data)
                shap_value = explainer(input_data)
                
                # ploting the individual prediction waterfall 
                st.subheader("Prediction Explanation")
                st.write("Feature contributions to the prediction:")
                
                fig , ax = plt.subplots()
                shap.plots.waterfall(shap_value[0,:, 1], show=False)
                st.pyplot(fig)
                
# Model Comaparison 
st.markdown("---")
st.subheader("Model comparison Table")
        
# x_test = joblib.load(os.path.join(base_path, "../models/x_test.pkl"))
# explainer = shap.Explainer(model, x_test)
# shap_values = explainer(x_test, check_additivity=False)

# st.subheader("Global Feature Importance")
# shap.plots.beeswarm(shap_values[..., 1], show=False)
# st.pyplot(plt.gcf())

# model_choice = st.selectbox("Choose a model to select",["Random Forest", "XGBoost", "Logicstic Regression"])

# model_map = {
#     "Random Forest" : "rf_model.pkl",
#     "XGBoost" : "xgb_model.pkl",
#     "Logicstic Regression" : "log_model.pkl"
# }
# model_file = os.path.join(base_path, f"../models/{model_map[model_choice]}")
# model = joblib.load(model_file)

model_scores = joblib.load(os.path.join(base_path, "../models/model_scores.pkl"))

# st.subheader("Model Performance Summary")
st.dataframe(model_scores.style.highlight_max(axis=0, color ="lightgreen"))



# --- footer ---
st.markdown("---")
st.markdown("Built with Streamlit | Project by Abhigyan Kumar Sinha ")
    










