import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ── Page Config ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

# ── Load the saved pipeline ────────────────────────────────────────────
@st.cache_resource   # loads model once, caches it for speed
def load_model():
    return joblib.load("diabetes_model.pkl")

model = load_model()

# ── Sidebar ────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    show_proba     = st.checkbox("Show Probability Chart", value=True)
    show_raw       = st.checkbox("Show Raw Input Values",  value=False)
    st.markdown("---")
    st.markdown("**About this App**")
    st.markdown("Trained on the **Pima Indians Diabetes** dataset (Kaggle).")
    st.markdown("Model: **Decision Tree** (depth=3, gini)")
    st.markdown("Test Accuracy: **87.7%** | AUC: **0.899**")

# ── Title ──────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Risk Predictor")
st.write("Enter patient clinical measurements and click **Predict** to assess diabetes risk.")
st.markdown("---")

# ── Input Form ─────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=17, value=3)
    glucose     = st.slider("Glucose Level (mg/dL)", 0, 200, 120)
    bp          = st.slider("Blood Pressure (mm Hg)", 0, 130, 70)

with col2:
    skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 30)
    insulin        = st.slider("Insulin (µU/mL)",     0, 850, 100)
    bmi            = st.number_input("BMI", min_value=10.0, max_value=70.0,
                                     value=30.5, step=0.1, format="%.1f")

with col3:
    dpf = st.number_input("Diabetes Pedigree Function",
                           min_value=0.0, max_value=2.5, value=0.5, step=0.01, format="%.3f")
    age = st.slider("Age (years)", 21, 81, 35)

st.markdown("---")

# ── Predict Button ─────────────────────────────────────────────────────
if st.button("🔍  Predict Diabetes Risk", type="primary", use_container_width=True):

    # Build input DataFrame — must match training feature order
    glucose_bmi = glucose * bmi   # engineered feature from Day 21!

    input_data = pd.DataFrame({
        "Pregnancies":            [pregnancies],
        "Glucose":                [glucose],
        "BloodPressure":          [bp],
        "SkinThickness":          [skin_thickness],
        "Insulin":                [insulin if insulin > 0 else 100],  # fill 0 with median
        "BMI":                    [bmi],
        "DiabetesPedigreeFunction":[dpf],
        "Age":                    [age],
        "Glu_BMI":            [glucose_bmi],
    })

    # Run prediction
    prediction   = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]

    # ── Display Result ──────────────────────────────────────────────
    st.markdown("## Results")
    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if prediction == 1:
            st.error("⚠️  Diabetes Risk Detected")
            st.metric("Diabetes Probability", f"{probabilities[1]*100:.1f}%",
                      delta="Above threshold")
        else:
            st.success("✅  No Diabetes Risk Detected")
            st.metric("No-Diabetes Probability", f"{probabilities[0]*100:.1f}%",
                      delta="Below threshold")

    with res_col2:
        if show_proba:
            proba_df = pd.DataFrame({
                "Outcome":     ["No Diabetes", "Diabetes"],
                "Probability": [round(probabilities[0]*100, 1),
                                round(probabilities[1]*100, 1)]
            }).set_index("Outcome")
            st.bar_chart(proba_df, use_container_width=True)

    # ── Show raw inputs if toggled ──────────────────────────────────
    if show_raw:
        st.markdown("### Input Values Sent to Model")
        st.dataframe(input_data, use_container_width=True)

    # ── Medical disclaimer ──────────────────────────────────────────
    st.info("**Disclaimer:** This tool is for educational demonstration only. "
            "Always consult a qualified healthcare professional for medical advice.")

# ── Footer ──────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Built with Streamlit | Model trained on Pima Indians Diabetes Dataset (Kaggle) | Day 22 — Data Science & AI Course")