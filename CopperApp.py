#!/usr/bin/env python
# coding: utf-8

# In[10]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the models and scaler
regressor = joblib.load("regressor_model.pkl")
classifier_won = joblib.load("classifier_won_model.pkl")
classifier_lost = joblib.load("classifier_lost_model.pkl")
scaler = joblib.load("scaler.pkl")

# Define expected input columns in correct order (21 features)
expected_columns = [
    'quantity tons', 'customer', 'country', 'item type', 'application',
    'thickness', 'width', 'product_ref', 'delivery date_year',
    'delivery date_month', 'delivery date_day',
    'status_lost', 'status_won',
    'item_date_year', 'item_date_month', 'item_date_day',
    'delivery period_year', 'delivery period_month', 'delivery period_day',
    'selling_price', 'status'  # Will drop before scaling
]

# Remove target columns from input features
input_features = [col for col in expected_columns if col not in ['selling_price', 'status']]

# Sidebar for user input
st.sidebar.header("Input Copper Deal Details")

def user_input_features():
    data = {
        'quantity tons': st.sidebar.number_input('Quantity (tons)', value=1000.0),
        'customer': st.sidebar.number_input('Customer ID', value=100),
        'country': st.sidebar.number_input('Country Code', value=15),
        'item type': st.sidebar.number_input('Item Type Code', value=0),
        'application': st.sidebar.number_input('Application Code', value=10),
        'thickness': st.sidebar.number_input('Thickness', value=0.5),
        'width': st.sidebar.number_input('Width', value=800),
        'product_ref': st.sidebar.number_input('Product Ref Code', value=611728),
        'delivery date': st.sidebar.number_input('delivery date', value=15),
        'status_Lost': st.sidebar.selectbox('Status Lost (1 = Lost, 0 = Not Lost)', [0, 1]),
        'status_Won': st.sidebar.selectbox('Status Won (1 = Won, 0 = Not Won)', [1, 0]),
        'item_date': st.sidebar.number_input('Item Date', value=2022)
        
    }
    return pd.DataFrame([data])

df_input = user_input_features()

# Initialize full input with zeros and update with user input
full_input = pd.DataFrame([np.zeros(len(input_features))], columns=input_features)
for col in df_input.columns:
    if col in full_input.columns:
        full_input.at[0, col] = df_input.at[0, col]

# Scale the input
scaled_input = scaler.transform(full_input)

# Prediction
selling_price_pred = regressor.predict(scaled_input)[0]
status_won_prob = classifier_won.predict_proba(scaled_input)[0][1]
status_lost_prob = classifier_lost.predict_proba(scaled_input)[0][1]

# Output
st.title("Copper Industry Forecasting App")

st.subheader("Predicted Selling Price")
st.write(f"₹ {selling_price_pred:,.2f}")

st.subheader("Deal Status Probabilities")
st.write(f"Probability of Status: **WON** → {status_won_prob:.2%}")
st.write(f"Probability of Status: **LOST** → {status_lost_prob:.2%}")

# Classification result
predicted_status = "WON" if status_won_prob > status_lost_prob else "LOST"
st.success(f"Predicted Deal Status: **{predicted_status}**")


# In[ ]:




