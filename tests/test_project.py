"""
tests/test_project.py
----------------------
Basic automated tests for the Industrial Machine Failure Prediction capstone.

Run with:
    pytest tests/test_project.py -v
"""
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from preprocessing import (  # noqa: E402
    ALL_FEATURES,
    LEAKAGE_COLUMNS,
    TARGET_COLUMN,
    load_raw_data,
    get_feature_target_split,
    build_model_pipeline,
)
from predict import load_pipeline, make_single_prediction  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

DATA_PATH = os.path.join(ROOT, "data", "ai4i2020.csv")
MODEL_PATH = os.path.join(ROOT, "models", "machine_failure_pipeline.pkl")


def test_dataset_loads():
    """1. The dataset file exists and can be loaded as a DataFrame."""
    assert os.path.exists(DATA_PATH), "ai4i2020.csv is missing from data/"
    df = load_raw_data(DATA_PATH)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_expected_columns_exist():
    """2. All required input feature columns exist after loading."""
    df = load_raw_data(DATA_PATH)
    for col in ALL_FEATURES:
        assert col in df.columns, f"Missing expected feature column: {col}"


def test_target_exists():
    """3. The target column exists and is binary (0/1)."""
    df = load_raw_data(DATA_PATH)
    assert TARGET_COLUMN in df.columns
    assert set(df[TARGET_COLUMN].unique()).issubset({0, 1})


def test_no_leakage_columns_in_features():
    """4. None of the known leakage columns ever end up in X."""
    df = load_raw_data(DATA_PATH)
    X, y = get_feature_target_split(df)
    for col in LEAKAGE_COLUMNS:
        assert col not in X.columns, f"Leakage column '{col}' leaked into features!"


def test_preprocessing_pipeline_fits_and_transforms():
    """5. The preprocessing + model pipeline can fit and predict on a tiny sample."""
    df = load_raw_data(DATA_PATH)
    X, y = get_feature_target_split(df)
    pipeline = build_model_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X, y)
    preds = pipeline.predict(X)
    assert len(preds) == len(X)
    assert set(preds).issubset({0, 1})


def test_saved_model_loads():
    """6. The saved trained pipeline file loads without error."""
    assert os.path.exists(MODEL_PATH), "Run src/train.py first to create the saved model."
    pipeline = load_pipeline(MODEL_PATH)
    assert pipeline is not None


def test_prediction_returns_valid_output():
    """7. A single prediction returns a valid class and probability."""
    pipeline = load_pipeline(MODEL_PATH)
    prediction, probability = make_single_prediction(
        pipeline,
        machine_type="L",
        air_temp=300.0,
        process_temp=310.0,
        rotational_speed=1400,
        torque=60.0,
        tool_wear=200,
    )
    assert prediction in (0, 1)
    if probability is not None:
        assert 0.0 <= probability <= 1.0


def test_streamlit_app_module_imports():
    """8. app.py can at least be imported (skipped if streamlit isn't installed
    in this environment, since Streamlit apps are normally run with the
    `streamlit run` command, not imported directly)."""
    pytest.importorskip("streamlit")
    import importlib.util

    app_path = os.path.join(ROOT, "app.py")
    spec = importlib.util.spec_from_file_location("app", app_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
