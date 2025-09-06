import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')

# Load saved encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)
st.title("Customer Churn Prediction")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
tenure = st.number_input("Tenure", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=1000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=10, value=1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)


# Prepare Input Data

input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary],
    'Gender': [gender],
    'Geography': [geography]
})


input_df['Gender'] = label_encoder_gender.transform(input_df['Gender'])
geo_encoded = onehot_encoder_geo.transform(input_df[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_df = pd.concat([input_df.drop('Geography', axis=1), geo_encoded_df], axis=1)


input_df = input_df[scaler.feature_names_in_]

# Scale features
input_scaled = scaler.transform(input_df)


# Predict Button

if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)
    probability = float(prediction[0][0])
    st.write(f"Prediction Probability of Churn: {probability:.2f}")
    st.write("Predicted Class:", "Churn" if probability > 0.5 else "Not Churn")

