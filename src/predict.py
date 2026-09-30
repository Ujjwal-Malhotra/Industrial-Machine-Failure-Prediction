"""
predict.py
----------
Small helper used by the Streamlit app (and by the tests) to load the saved
pipeline and turn one set of user inputs into a prediction.

WHY a separate file: keeping the "load model + predict" logic in one place
means both app.py and the automated tests use exactly the same code path,
so we are always testing what the app actually does.
"""
from __future__ import annotations

import joblib
import pandas as pd

from preprocessing import ALL_FEATURES

DEFAULT_MODEL_PATH = "models/machine_failure_pipeline.pkl"


def load_pipeline(model_path: str = DEFAULT_MODEL_PATH):
    """Load the saved scikit-learn Pipeline (preprocessing + model)."""
    return joblib.load(model_path)


def make_single_prediction(pipeline, machine_type: str, air_temp: float,
                            process_temp: float, rotational_speed: float,
                            torque: float, tool_wear: float):
    """Build a one-row DataFrame in the exact column order the pipeline
    expects, then return (predicted_class, failure_probability)."""
    row = pd.DataFrame(
        [{
            "Air temperature [K]": air_temp,
            "Process temperature [K]": process_temp,
            "Rotational speed [rpm]": rotational_speed,
            "Torque [Nm]": torque,
            "Tool wear [min]": tool_wear,
            "Type": machine_type,
        }]
    )[ALL_FEATURES]

    prediction = int(pipeline.predict(row)[0])
    probability = None
    if hasattr(pipeline, "predict_proba"):
        probability = float(pipeline.predict_proba(row)[0, 1])
    return prediction, probability
