import streamlit as st
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8080/predict"

st.set_page_config(page_title="🩺 Cirrhosis Detection App", layout="centered")
st.title("🩸 Cirrhosis Detection")
st.markdown("Enter patient details below to predict cirrhosis severity.")

with st.form("patient_form"):
    col1, col2 = st.columns(2)

    with col1:
        Sex = st.selectbox("Sex", ["F", "M"])
        Drug = st.selectbox("Drug", ["Placebo", "D-penicillamine"])
        Ascites = st.selectbox("Ascites", ["N", "Y"])
        Hepatomegaly = st.selectbox("Hepatomegaly", ["N", "Y"])
        Spiders = st.selectbox("Spiders", ["N", "Y"])
        Edema = st.selectbox("Edema", ["N", "Y", "S"])
        Status = st.selectbox("Status", ["C", "D", "CL"])

    with col2:
        Age = st.number_input("Age (years)", min_value=10.0, max_value=100.0, value=52.0)
        Bilirubin = st.number_input("Bilirubin (mg/dL)", min_value=0.0, max_value=20.0, value=1.4)
        Albumin = st.number_input("Albumin (g/dL)", min_value=0.0, max_value=6.0, value=3.2)
        Alk_Phos = st.number_input("Alkaline Phosphatase", min_value=0.0, max_value=1000.0, value=120.5)
        SGOT = st.number_input("SGOT (AST)", min_value=0.0, max_value=500.0, value=45.8)
        Platelets = st.number_input("Platelets (x10⁹/L)", min_value=0.0, max_value=1000.0, value=210.0)
        Prothrombin = st.number_input("Prothrombin (s)", min_value=0.0, max_value=30.0, value=11.3)

    submitted = st.form_submit_button("🔍 Predict")

if submitted:
    payload = {
        "Sex": Sex,
        "Drug": Drug,
        "Ascites": Ascites,
        "Hepatomegaly": Hepatomegaly,
        "Spiders": Spiders,
        "Edema": Edema,
        "Status": Status,
        "Age": Age,
        "Bilirubin": Bilirubin,
        "Albumin": Albumin,
        "Alk_Phos": Alk_Phos,
        "SGOT": SGOT,
        "Platelets": Platelets,
        "Prothrombin": Prothrombin
    }

    try:
        with st.spinner("Predicting..."):
            response = requests.post(API_URL, json=payload)
            result = response.json()
        if "prediction" in result:
            st.success(f"**Prediction:** {result['prediction']}")
        else:
            st.error("Unexpected response format.")
            st.json(result)
            
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to FastAPI backend. Make sure it's running on port 8080.")
