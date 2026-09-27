import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")
RESULT_PATH = os.path.join(BASE_DIR, "results", "model_comparison.csv")

FEATURES = [
    "Nitrogen", "Phosphorus", "Potassium",
    "Temperature", "Humidity", "pH_Value", "Rainfall"
]

st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Crop Recommendation System")
st.write(
    "Enter soil and environmental conditions to get a machine-learning "
    "based crop recommendation."
)

if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
    st.error("Model files are missing.")
    st.info("Open the VS Code terminal and run: python train_model.py")
    st.stop()

model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

st.subheader("Enter Input Parameters")

col1, col2 = st.columns(2)

with col1:
    nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, max_value=150.0, value=50.0)
    phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, max_value=150.0, value=50.0)
    potassium = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=50.0)
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=25.0)

with col2:
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=70.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=500.0, value=150.0)

if st.button("🌾 Recommend Crop", use_container_width=True):
    input_data = pd.DataFrame([[
        nitrogen, phosphorus, potassium,
        temperature, humidity, ph, rainfall
    ]], columns=FEATURES)

    pred_encoded = model.predict(input_data)[0]
    crop = encoder.inverse_transform([pred_encoded])[0]

    st.success(f"Recommended Crop: **{crop}**")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        confidence = float(np.max(probabilities) * 100)
        st.metric("Model Confidence", f"{confidence:.2f}%")

        top_idx = np.argsort(probabilities)[::-1][:3]
        top_rows = pd.DataFrame({
            "Crop": encoder.inverse_transform(top_idx),
            "Probability": [f"{probabilities[i] * 100:.2f}%" for i in top_idx]
        })
        st.write("Top 3 model predictions:")
        st.table(top_rows)

st.divider()

with st.expander("About this project"):
    st.write(
        "The system compares Logistic Regression, Decision Tree, Random Forest "
        "and SVM during training. Random Forest is used by the application as "
        "the final classifier."
    )

if os.path.exists(RESULT_PATH):
    with st.expander("Model Comparison"):
        results = pd.read_csv(RESULT_PATH)
        st.dataframe(results, use_container_width=True)
