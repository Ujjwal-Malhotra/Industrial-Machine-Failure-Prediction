"""
train.py
--------
Trains the required progression of beginner-level models on the AI4I 2020
dataset, evaluates them with imbalance-aware metrics, picks a final model,
and saves the complete preprocessing + model pipeline with joblib.

Run:
    python src/train.py --data data/ai4i2020.csv --out models/machine_failure_pipeline.pkl
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from preprocessing import (
    RANDOM_STATE,
    build_model_pipeline,
    get_feature_target_split,
    load_raw_data,
)


def evaluate(name, pipeline, X_test, y_test, results):
    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline, "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        y_proba = None

    row = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_proba) if y_proba is not None and len(set(y_test)) > 1 else float("nan"),
    }
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(row)
    print("Confusion matrix [[TN FP][FN TP]]:")
    print(cm)
    results.append(row)
    return cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ai4i2020.csv")
    parser.add_argument("--out", default="models/machine_failure_pipeline.pkl")
    parser.add_argument("--metrics_out", default="models/metrics_summary.csv")
    args = parser.parse_args()

    df = load_raw_data(args.data)
    X, y = get_feature_target_split(df)

    print(f"Loaded {len(df)} rows. Failure rate: {y.mean() * 100:.2f}%")

    # Stratified split keeps the (small) failure class proportionally
    # represented in both the train and test sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    results = []
    confusions = {}

    # 1. Dummy baseline - always predicts the majority class.
    # WHY: this tells us what accuracy a "do-nothing" model would already
    # get, purely because failures are rare. Any real model must beat this
    # in a meaningful metric (recall/F1), not just in accuracy.
    dummy = build_model_pipeline(DummyClassifier(strategy="most_frequent"))
    dummy.fit(X_train, y_train)
    confusions["Dummy Baseline"] = evaluate("Dummy Baseline", dummy, X_test, y_test, results)

    # 2. Plain Logistic Regression
    logreg = build_model_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    logreg.fit(X_train, y_train)
    confusions["Logistic Regression"] = evaluate("Logistic Regression", logreg, X_test, y_test, results)

    # 3. Class-weighted Logistic Regression
    # WHY: class_weight="balanced" tells the model to pay proportionally
    # more attention to the rare failure class instead of ignoring it.
    logreg_bal = build_model_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
    )
    logreg_bal.fit(X_train, y_train)
    confusions["Weighted Logistic Regression"] = evaluate(
        "Weighted Logistic Regression", logreg_bal, X_test, y_test, results
    )

    # 4. Decision Tree
    tree = build_model_pipeline(
        DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE)
    )
    tree.fit(X_train, y_train)
    confusions["Decision Tree"] = evaluate("Decision Tree", tree, X_test, y_test, results)

    # 5. Random Forest (kept simple - modest number of shallow trees)
    forest = build_model_pipeline(
        RandomForestClassifier(
            n_estimators=200, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
        )
    )
    forest.fit(X_train, y_train)
    confusions["Random Forest"] = evaluate("Random Forest", forest, X_test, y_test, results)

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    results_df.to_csv(args.metrics_out, index=False)
    print("\n=== FINAL COMPARISON TABLE (actual results) ===")
    print(results_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Final model selection
    # WHY: we do NOT pick the model with the highest accuracy, because
    # accuracy is misleading on an imbalanced dataset. We pick the model
    # with the best F1-score on the failure class, since F1 balances
    # precision (avoiding false alarms) and recall (catching real failures).
    # ------------------------------------------------------------------
    # The Dummy Baseline is only ever used as a reference point (it is not a
    # real candidate model), so it is excluded from the "best model" search.
    candidates_df = results_df[results_df["Model"] != "Dummy Baseline"]
    candidates_df = candidates_df.sort_values(["F1", "Recall"], ascending=False)
    best_row = candidates_df.iloc[0]
    best_name = best_row["Model"]
    pipelines = {
        "Dummy Baseline": dummy,
        "Logistic Regression": logreg,
        "Weighted Logistic Regression": logreg_bal,
        "Decision Tree": tree,
        "Random Forest": forest,
    }
    best_pipeline = pipelines[best_name]

    print(f"\nSelected final model: {best_name} (highest F1 on the failure class)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(best_pipeline, args.out)
    print(f"Saved trained pipeline to {args.out}")

    with open(os.path.join(os.path.dirname(args.out), "selected_model.json"), "w") as f:
        json.dump({"selected_model": best_name}, f, indent=2)


if __name__ == "__main__":
    main()
