# Industrial Machine Failure Prediction using Machine Learning

**Learn Depth Academy — Track 1 Final Capstone**
**Problem 21 — Industrial Machine Failure Prediction**

> ⚠️ **Data limitation notice (please read):** This deliverable was built inside a
> sandboxed development environment with **no internet access**. Only the
> **first 100 real rows** of the 10,000-row AI4I 2020 dataset could be verified
> and downloaded automatically inside that sandbox. Everything in this repo —
> code, notebook, saved model, metrics — was **actually run**, and every number
> reported is real (nothing fabricated). But because only 100 of 10,000 rows
> (about 3 real failures) were available, the reported metrics are **illustrative,
> not final**. Step 1 in "Installation" below tells you how to drop in the full
> file — after that, every script and the notebook already work end-to-end and
> will produce your real final numbers with **no code changes needed**.

## 1. Problem Statement
"Predict whether a machine is likely to experience a failure within a defined
operating window."

## 2. Objective
Build an end-to-end, zero-cost, beginner-level machine learning solution that
loads and explores a public dataset, preprocesses it safely, trains and compares
several foundational classification models, handles class imbalance properly,
interprets the results, and ships a small Streamlit app for making predictions —
all while being fully reproducible and well documented.

## 3. Why Machine Failure Prediction Matters
Unplanned industrial machine failures cause costly downtime, wasted materials,
missed deadlines, and sometimes safety risks. Predictive maintenance uses data
(sensor readings, process parameters) to flag machines that are *likely* to fail
soon, so maintenance teams can act *before* a breakdown instead of *after* one.

## 4. Dataset
**AI4I 2020 Predictive Maintenance Dataset**
Source: UCI Machine Learning Repository
https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

- 10,000 rows, 14 original columns
- Target: `Machine failure` (binary: 0 = no failure, 1 = failure)
- Severely imbalanced (~3.4% failure rate in the full dataset)

### 5. Dataset Source / Citation
> AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine
> Learning Repository. https://doi.org/10.24432/C5HS5C
>
> S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance
> Applications," 2020 Third International Conference on Artificial Intelligence
> for Industries (AI4I), 2020, pp. 69-74, doi: 10.1109/AI4I49448.2020.00023.
>
> License: Creative Commons Attribution 4.0 International (CC BY 4.0).

### 6. Dataset Limitation: Synthetic Benchmark
AI4I 2020 is a **publicly available synthetic dataset** designed by its author to
*reflect* real predictive-maintenance conditions found in industry — it is
**not** raw sensor data captured from an actual running factory. We treat the
underlying problem (predicting machine failure) as a genuine, real industrial
problem, while being explicit that this specific dataset is a synthetic
benchmark used for learning and research, not a production data feed.

### 7. Learn Depth Problem ↔ Dataset Mapping
The Learn Depth problem statement references generic sensor concepts
(temperature, vibration, pressure, runtime, maintenance gap). AI4I 2020 does not
literally contain columns called "vibration" or "maintenance gap" — instead it
operationalizes the same real-world failure-prediction problem through the
variables it actually provides:

| Learn Depth generic concept | AI4I 2020 actual column      |
|---|---|
| Temperature                 | `Air temperature [K]`, `Process temperature [K]` |
| (mechanical stress proxy)   | `Torque [Nm]`, `Rotational speed [rpm]` |
| Runtime / wear              | `Tool wear [min]` |
| Machine/product context     | `Type` (L/M/H quality variant) |
| Failure outcome             | `Machine failure` (target) |

## 8. Features Used
- `Type` (categorical: L / M / H)
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

## 9. Features Excluded and Why
| Column | Why excluded |
|---|---|
| `UDI` | Row identifier with no physical meaning — using it would let the model memorize rows instead of learning patterns. |
| `Product ID` | Also just an identifier (with the Type letter embedded in it). |
| `TWF`, `HDF`, `PWF`, `OSF`, `RNF` | These are the individual failure-mode flags whose logical OR **is** the `Machine failure` target. Using them as input features would be **data leakage** — the model would essentially be given the answer. They are only used for EDA (to understand *why* machines failed), never as model inputs. |

## 10. Data Preprocessing
- Train/test split performed **before** any preprocessing is fitted (80/20,
  stratified on the target, `random_state=42`).
- `StandardScaler` on the 5 numeric features (Logistic Regression is scale-sensitive).
- `OneHotEncoder` on `Type`.
- Both wrapped in a single scikit-learn `Pipeline` + `ColumnTransformer`, fitted
  only on the training data, so the exact same transformation is applied at
  prediction time (in tests and in the Streamlit app) with zero risk of
  train/test leakage.

## 11. EDA Summary
- No missing values, no duplicate rows found.
- Target is heavily imbalanced toward "no failure".
- Some numeric features (e.g. torque × rotational speed) have physically
  meaningful "unusual" combinations tied directly to specific failure modes
  (e.g. Power Failure), so outliers were **not** blindly removed.
- Full plots and discussion are in `notebooks/machine_failure_prediction.ipynb`.

## 12. Models Tested
1. `DummyClassifier` (most-frequent baseline)
2. Logistic Regression
3. Class-weighted Logistic Regression (`class_weight="balanced"`)
4. Decision Tree (`class_weight="balanced"`)
5. Random Forest (`class_weight="balanced"`, kept simple: 200 shallow trees)

No deep learning, no XGBoost, no heavy hyperparameter tuning — this is a
foundational-ML capstone by design.

## 13. Class Imbalance
Failures are rare. We:
- Never rely on accuracy alone.
- Use `class_weight="balanced"` as the first, simplest imbalance strategy.
- Did **not** add SMOTE: with the small real sample available in this sandbox it
  would not be meaningful to demonstrate honestly, and it is not required when
  `class_weight="balanced"` already gives a working, explainable baseline. (If
  you want to try it on the full dataset, `imbalanced-learn`'s `SMOTE` inside a
  cross-validated pipeline — never before the train/test split — is the
  recommended next step; see Future Scope.)

## 14. Evaluation Metrics
Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC are
reported for every model.

## 15. Actual Final Results
*(computed on the 100-row real sample available in this sandbox — see the data
limitation notice at the top; re-run `python src/train.py` after replacing
`data/ai4i2020.csv` with the full 10,000-row file for your real final numbers)*

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Dummy Baseline | 0.95 | 0.00 | 0.00 | 0.00 | 0.50 |
| Logistic Regression | 0.95 | 0.00 | 0.00 | 0.00 | **0.89** |
| Weighted Logistic Regression | 0.95 | 0.00 | 0.00 | 0.00 | **0.89** |
| Decision Tree | 0.90 | 0.00 | 0.00 | 0.00 | 0.47 |
| Random Forest | 0.95 | 0.00 | 0.00 | 0.00 | 0.34 |

With only one failing machine in this tiny test split, Precision/Recall/F1 all
collapse to 0 whenever that single case is missed — a direct, real illustration
of why class imbalance matters and why 100 rows is not enough to draw real
conclusions. ROC-AUC (which does not depend on a single decision threshold)
still shows a real, meaningful signal: both Logistic Regression variants reach
0.89, clearly ahead of the tree-based models on this tiny sample.

## 16. Final Model Selection Reasoning
Selection rule: **highest F1-score on the failure class**, excluding the Dummy
Baseline (which is a reference point, not a real candidate), with ROC-AUC as a
tie-breaker. On the available sample this selects **Logistic Regression**. This
rule is written into `src/train.py` and will automatically select whichever
model performs best once you run it on the full dataset.

## 17. Project Structure
```
Industrial-Machine-Failure-Prediction/
├── data/
│   └── ai4i2020.csv            # replace with the full 10,000-row file (see below)
├── notebooks/
│   └── machine_failure_prediction.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
├── models/
│   ├── machine_failure_pipeline.pkl
│   └── metrics_summary.csv
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── report/
│   └── technical_paper.md
├── presentation/
│   └── presentation.md
├── VIVA_QA.md
└── tests/
    └── test_project.py
```

## 18. Installation
```
python -m venv .venv
```
**Activate on Windows:**
```
.venv\Scripts\activate
```
**Activate on macOS/Linux:**
```
source .venv/bin/activate
```
**Install dependencies:**
```
pip install -r requirements.txt
```

### Getting the full dataset (do this before your final run)
1. Go to https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset
2. Click **Download**, unzip it, and copy `ai4i2020.csv` into the `data/` folder
   (overwrite the sample file that is already there).
3. Re-run training (see below) — no code changes needed.

## 19. Running the Notebook
```
jupyter notebook notebooks/machine_failure_prediction.ipynb
```
Run all cells top to bottom (Kernel → Restart & Run All). It will re-save the
model pipeline to `models/machine_failure_pipeline.pkl`.

You can also retrain from the command line:
```
python src/train.py --data data/ai4i2020.csv --out models/machine_failure_pipeline.pkl
```

## 20. Running Streamlit
```
streamlit run app.py
```
This opens a local web page where you can enter Type, Air temperature, Process
temperature, Rotational speed, Torque, and Tool wear, and get a prediction plus
failure probability.

## 21. Example Application Workflow
1. Train (or re-train) the model: `python src/train.py`
2. Launch the app: `streamlit run app.py`
3. Enter operating conditions, e.g. Type=`L`, Air temp=`302`, Process temp=`311`,
   Speed=`1400`, Torque=`60`, Tool wear=`210`.
4. Click **Predict** → app shows "Machine Failure" / "No Machine Failure" and a
   failure probability percentage.

## 22. Limitations
- **Sandbox data limitation:** only 100 of 10,000 rows were available while
  building this deliverable (see the notice at the top); results must be
  re-validated on the full file.
- The dataset is synthetic, not real factory sensor data.
- Models are intentionally simple (no deep learning, no heavy tuning) by design
  of the capstone brief.
- No temporal/sequence modeling — each row is treated as an independent snapshot.

## 23. Future Scope
- Re-run everything on the full 10,000-row dataset for real, final metrics.
- Try SMOTE (inside cross-validation folds only) and compare against
  `class_weight="balanced"`.
- Add simple hyperparameter tuning (e.g. `GridSearchCV`) once the pipeline is
  validated on full data.
- Explore probability calibration for more trustworthy failure probabilities.

## 24. AI Assistance Disclosure
This project (code, notebook, documentation, technical paper, presentation, and
viva questions) was built with the assistance of an AI coding assistant (Claude),
under human direction, as part of a Learn Depth Academy capstone. Data loading,
model training, metric computation, and automated tests were actually executed
in a sandboxed environment; the dataset limitation is disclosed above.

## 25. References
1. AI4I 2020 Predictive Maintenance Dataset [Dataset]. (2020). UCI Machine
   Learning Repository. https://doi.org/10.24432/C5HS5C
2. S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance
   Applications," 2020 Third International Conference on Artificial
   Intelligence for Industries (AI4I), 2020, pp. 69-74.
3. scikit-learn documentation: https://scikit-learn.org/stable/documentation.html
4. Streamlit documentation: https://docs.streamlit.io/

See also `VIVA_QA.md`, `report/technical_paper.md`, and `presentation/presentation.md`.
