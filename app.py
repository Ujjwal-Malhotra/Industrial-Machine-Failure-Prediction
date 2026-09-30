"""
app.py
------
Simple Streamlit app for the Industrial Machine Failure Prediction capstone.

Run with:
    streamlit run app.py

This app loads the ALREADY-TRAINED pipeline saved at
models/machine_failure_pipeline.pkl. It does NOT retrain anything, so it
starts instantly and always uses the same model that was evaluated in the
notebook / technical paper.
"""
import sys
import os

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from predict import load_pipeline, make_single_prediction  # noqa: E402

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "machine_failure_pipeline.pkl")

st.set_page_config(page_title="Machine Failure Predictor", page_icon="🛠️")

st.title("🛠️ Industrial Machine Failure Predictor")
st.caption(
    "Educational capstone project — AI4I 2020 Predictive Maintenance dataset "
    "(a public synthetic benchmark, not real factory data). This is a "
    "learning tool, NOT a certified maintenance system."
)

# ---------------------------------------------------------------------------
# Load the trained pipeline once
# ---------------------------------------------------------------------------
@st.cache_resource
def get_pipeline():
    return load_pipeline(MODEL_PATH)


try:
    pipeline = get_pipeline()
except FileNotFoundError:
    st.error(
        "No trained model found at models/machine_failure_pipeline.pkl.\n\n"
        "Run `python src/train.py` first to train and save the pipeline."
    )
    st.stop()

st.subheader("Enter the machine's current operating conditions")

col1, col2 = st.columns(2)

with col1:
    machine_type = st.selectbox("Product quality type", ["L", "M", "H"], index=0)
    air_temp = st.number_input(
        "Air temperature [K]", min_value=290.0, max_value=310.0, value=300.0, step=0.1
    )
    process_temp = st.number_input(
        "Process temperature [K]", min_value=295.0, max_value=315.0, value=310.0, step=0.1
    )

with col2:
    rotational_speed = st.number_input(
        "Rotational speed [rpm]", min_value=1100, max_value=3000, value=1500, step=10
    )
    torque = st.number_input(
        "Torque [Nm]", min_value=0.0, max_value=80.0, value=40.0, step=0.5
    )
    tool_wear = st.number_input(
        "Tool wear [min]", min_value=0, max_value=260, value=100, step=1
    )

st.caption(
    "The default min/max ranges above roughly match the values seen in the "
    "AI4I 2020 dataset. Values far outside these ranges are still accepted, "
    "but the model has not seen data like that during training."
)

if st.button("Predict"):
    prediction, probability = make_single_prediction(
        pipeline,
        machine_type=machine_type,
        air_temp=air_temp,
        process_temp=process_temp,
        rotational_speed=rotational_speed,
        torque=torque,
        tool_wear=tool_wear,
    )

    if prediction == 1:
        st.error("⚠️ Prediction: **Machine Failure**")
    else:
        st.success("✅ Prediction: **No Machine Failure**")

    if probability is not None:
        st.metric("Estimated failure probability", f"{probability * 100:.1f}%")
        st.progress(min(max(probability, 0.0), 1.0))

    st.markdown(
        """
**How to read this result:** the model looks at the operating conditions you
entered and compares them to patterns it learned from the training data.
A higher failure probability generally means the combination of high torque,
low rotational speed, high tool wear, and/or a large temperature difference
looked similar to conditions that led to failure in the training data.
This is a statistical association, **not proof of a physical cause**.
"""
    )

st.divider()
st.caption(
    "Educational capstone project. Dataset: AI4I 2020 Predictive Maintenance "
    "Dataset, UCI Machine Learning Repository (CC BY 4.0)."
)
