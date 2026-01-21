import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model/energy_model.pkl")

st.set_page_config(page_title="EV Energy Predictor")

st.title("🔋 EV Charging Energy Consumption Predictor")
st.write("Predicts whether an EV charging session will consume **Low, Medium, or High energy**.")

st.sidebar.header("Enter Charging Session Details")

# User inputs
vehicle_type = st.sidebar.selectbox("Vehicle Type", ["Car", "Bike", "Bus"])
payment_method = st.sidebar.selectbox("Payment Method", ["UPI", "Card", "Wallet"])
duration = st.sidebar.number_input("Charging Duration (minutes)", min_value=1.0)
cost = st.sidebar.number_input("Cost (INR)", min_value=1.0)

# Manual encoding (must match training logic)
vehicle_map = {"Car": 0, "Bike": 1, "Bus": 2}
payment_map = {"UPI": 0, "Card": 1, "Wallet": 2}

vehicle_encoded = vehicle_map[vehicle_type]
payment_encoded = payment_map[payment_method]

# Predict
if st.button("Predict Energy Consumption"):
    input_df = pd.DataFrame({
        'Vehicle_Type': [vehicle_encoded],
        'Charging_Duration_Min': [duration],
        'Cost_INR': [cost],
        'Payment_Method': [payment_encoded]
    })

    prediction = model.predict(input_df)[0]

    st.subheader("🔮 Prediction Result")

    if prediction == "High":
        st.error("⚡ High Energy Consumption")
    elif prediction == "Medium":
        st.warning("🔋 Medium Energy Consumption")
    else:
        st.success("✅ Low Energy Consumption")
