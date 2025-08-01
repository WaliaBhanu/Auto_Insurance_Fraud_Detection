# streamlit_app.py

import streamlit as st
import pandas as pd
import pickle

# Load trained pipeline
with open("random_forest_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

st.title("Insurance Fraud Prediction App")

st.write("Enter claim details to predict whether it's fraudulent or not:")

# Define input fields (use the same columns as training data)
input_data = {}

input_data["policy_state"] = st.selectbox("Policy State", ["OH", "IN", "IL"])
input_data["policy_csl"] = st.selectbox("Policy CSL", ["250/500", "100/300", "500/1000"])
input_data["incident_type"] = st.selectbox("Incident Type", ["Single Vehicle Collision", "Multi-vehicle Collision", "Parked Car", "Vehicle Theft"])
input_data["insured_occupation"] = st.selectbox("Insured Occupation", ["craft-repair", "sales", "exec-managerial", "tech-support", "other"])
input_data["insured_relationship"] = st.selectbox("Insured Relationship", ["own-child", "husband", "wife", "not-in-family"])
input_data["authorities_contacted"] = st.selectbox("Authorities Contacted", ["Police", "Fire", "Other", "None"])
input_data["incident_state"] = st.selectbox("Incident State", ["NY", "SC", "VA", "OH"])
input_data["incident_city"] = st.selectbox("Incident City", ["Columbus", "Arlington", "Riverwood", "Springfield", "Northbrook"])

input_data["insured_sex"] = st.selectbox("Insured Sex", ["MALE", "FEMALE"])
input_data["property_damage"] = st.selectbox("Property Damage", ["YES", "NO"])
input_data["police_report_available"] = st.selectbox("Police Report Available", ["YES", "NO"])
input_data["collision_type"] = st.selectbox("Collision Type", ["Rear Collision", "Side Collision", "Front Collision"])

input_data["auto_make"] = st.selectbox("Auto Make", ["Toyota", "Ford", "BMW", "Chevrolet"])
input_data["auto_model"] = st.selectbox("Auto Model", ["Corolla", "Accord", "Civic", "Camry"])
input_data["insured_zip"] = st.text_input("Insured ZIP", "10001")

input_data["insured_education_level"] = st.selectbox("Education Level", ["JD", "High School", "Associate", "MD", "Masters", "PhD", "College"])
input_data["incident_severity"] = st.selectbox("Incident Severity", ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"])

input_data["policy_deductable"] = st.number_input("Policy Deductible", value=500)
input_data["policy_annual_premium"] = st.number_input("Annual Premium", value=1000.0)
input_data["total_claim_amount"] = st.number_input("Total Claim Amount", value=5000.0)
input_data["injury_claim"] = st.number_input("Injury Claim", value=2000.0)
input_data["property_claim"] = st.number_input("Property Claim", value=1500.0)
input_data["vehicle_claim"] = st.number_input("Vehicle Claim", value=2500.0)
input_data["umbrella_limit"] = st.number_input("Umbrella Limit", value=0.0)
input_data["capital-gains"] = st.number_input("Capital Gains", value=0.0)
input_data["capital-loss"] = st.number_input("Capital Loss", value=0.0)

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Predict
if st.button("Predict Fraud"):
    prediction = pipeline.predict(input_df)[0]
    proba = pipeline.predict_proba(input_df)[0][1]

    st.subheader("Prediction:")
    st.write("✅ **Fraud Detected**" if prediction == 1 else "❌ **Not Fraudulent**")

    st.subheader("Fraud Probability:")
    st.write(f"{proba:.2%}")
