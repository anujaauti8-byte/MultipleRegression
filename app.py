
import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("🏠 Boston House Price Prediction")

st.write("Enter the house details below:")

# Input fields
CRIM = st.number_input("CRIM", value=0.0)
ZN = st.number_input("ZN", value=0.0)
INDUS = st.number_input("INDUS", value=0.0)
CHAS = st.number_input("CHAS", value=0)
NOX = st.number_input("NOX", value=0.0)
RM = st.number_input("RM", value=0.0)
AGE = st.number_input("AGE", value=0.0)
DIS = st.number_input("DIS", value=0.0)
RAD = st.number_input("RAD", value=0)
TAX = st.number_input("TAX", value=0.0)
PTRATIO = st.number_input("PTRATIO", value=0.0)
B = st.number_input("B", value=0.0)
LSTAT = st.number_input("LSTAT", value=0.0)

if st.button("Predict Price"):

    data = pd.DataFrame(
        [[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX,
          PTRATIO, B, LSTAT]],
        columns=[
            'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
            'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
        ]
    )

    # Scale input data
    scaled_data = scaler.transform(data)

    # Predict
    prediction = model.predict(scaled_data)

    st.success(f"Predicted House Price: ${prediction[0]:,.2f}")

