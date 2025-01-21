import streamlit as st
import pickle
import numpy as np
import xgboost as xgb
import pandas as pd

# Load the pre-trained XGBoost model
model_path = '/mnt/data/xgb_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Streamlit app layout
st.title("Disease Prediction Model")
st.write("Enter the following features to get a disease prediction:")

# Define input fields for each feature (excluding 'category' and 'sex')
age = st.number_input("age", min_value=0, max_value=120, step=1)
albumin = st.number_input("albumin", min_value=0.0)
alkaline_phosphatase = st.number_input("alkaline Phosphatase", min_value=0.0)
alanine_aminotransferase = st.number_input("alanine Aminotransferase", min_value=0.0)
aspartate_aminotransferase = st.number_input("aspartate Aminotransferase", min_value=0.0)
bilirubin = st.number_input("bailirubin", min_value=0.0)
cholinesterase = st.number_input("cholinesterase", min_value=0.0)
cholesterol = st.number_input("cholesterol", min_value=0.0)
creatinina = st.number_input("creatinina", min_value=0.0)
gamma_glutamyl_transferase = st.number_input("gamma-Glutamyl Transferase", min_value=0.0)
protein = st.number_input("Protein", min_value=0.0)

# Button for prediction
if st.button("Predict"):
    # Prepare the features for prediction (excluding 'category' and 'sex')
    features = np.array([age, albumin, alkaline_phosphatase, alanine_aminotransferase,
                         aspartate_aminotransferase, bilirubin, cholinesterase, cholesterol,
                         creatinina, gamma_glutamyl_transferase, protein]).reshape(1, -1)
    dmatrix = xgb.DMatrix(features)  # Prepare for XGBoost

    # Make prediction
    prediction = model.predict(dmatrix)
    result = prediction[0]  # Extract single prediction

    # Display the result
    st.write("Prediction:", result)
