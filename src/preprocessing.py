"""
preprocessing.py
-----------------
Beginner-friendly preprocessing utilities for the Industrial Machine Failure
Prediction capstone (AI4I 2020 dataset).

WHAT this file does:
    - Loads the raw AI4I 2020 csv file.
    - Selects only the SAFE input features (no ID columns, no failure-mode
      columns) to avoid data leakage.
    - Builds a scikit-learn ColumnTransformer that scales numeric columns and
      one-hot encodes the categorical "Type" column.

WHY it is written this way:
    - Data leakage means giving the model information it would not have at
      prediction time, or information that is only known AFTER a failure has
      already happened. UDI / Product ID are just row identifiers with no
      real physical meaning, and TWF/HDF/PWF/OSF/RNF are the individual
      failure-mode flags that make up "Machine failure" itself, so including
      them would let the model "cheat" instead of learning from the real
      operating conditions (temperature, speed, torque, tool wear).
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ---------------------------------------------------------------------------
# Column configuration
# ---------------------------------------------------------------------------

# Safe numeric input features (real operating measurements)
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

# Safe categorical input feature
CATEGORICAL_FEATURES = ["Type"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# The prediction target
TARGET_COLUMN = "Machine failure"

# Columns that must NEVER be used as model inputs (leakage / identifiers)
LEAKAGE_COLUMNS = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]

RANDOM_STATE = 42


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load the raw AI4I 2020 csv file exactly as downloaded."""
    df = pd.read_csv(csv_path)

    # Some public mirrors of this dataset use the column names
    # "Target" / "Failure Type" instead of the original UCI column names
    # "Machine failure" / TWF,HDF,PWF,OSF,RNF. We normalise here so the rest
    # of the project can always rely on the original UCI column names.
    if "Target" in df.columns and TARGET_COLUMN not in df.columns:
        df = df.rename(columns={"Target": TARGET_COLUMN})

    if "Failure Type" in df.columns and not set(
        ["TWF", "HDF", "PWF", "OSF", "RNF"]
    ).issubset(df.columns):
        # Reconstruct the individual failure-mode flags from the single
        # "Failure Type" column purely for EDA / documentation purposes.
        # These reconstructed flags are NEVER used as model features.
        for mode, flag in [
            ("Tool Wear Failure", "TWF"),
            ("Heat Dissipation Failure", "HDF"),
            ("Power Failure", "PWF"),
            ("Overstrain Failure", "OSF"),
            ("Random Failures", "RNF"),
        ]:
            df[flag] = (df["Failure Type"] == mode).astype(int)

    return df


def get_feature_target_split(df: pd.DataFrame):
    """Return (X, y) using ONLY the leakage-safe input features."""
    X = df[ALL_FEATURES].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build the ColumnTransformer used inside every model Pipeline.

    Numeric columns are standardised (mean 0, std 1) because Logistic
    Regression is sensitive to feature scale. Tree-based models do not
    strictly need scaling, but using the same preprocessing for every model
    keeps the project simple and the comparison fair.
    """
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
    return preprocessor


def build_model_pipeline(model) -> Pipeline:
    """Wrap any scikit-learn classifier together with the preprocessor."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )
