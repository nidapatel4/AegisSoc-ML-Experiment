from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "data" / "UNSW_NB15_training-set.csv"
TEST_PATH = BASE_DIR / "data" / "UNSW_NB15_testing-set.csv"

MODEL_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================
# RANDOM FOREST ARTIFACTS
# ============================================================

RF_MODEL_PATH = MODEL_DIR / "unsw_random_forest.joblib"
RF_PREPROCESSOR_PATH = (
    MODEL_DIR / "unsw_random_forest_preprocessor.joblib"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

# IF operating point
TARGET_FPR = 0.05

# RF threshold selected using validation data
RF_THRESHOLD = 0.65


# ============================================================
# ISOLATION FOREST CONFIG
# Same as UNSW baseline
# ============================================================

N_ESTIMATORS = 800
MAX_SAMPLES = 1.0
MAX_FEATURES = 0.7
CONTAMINATION = 0.01


# ============================================================
# COLUMNS
# ============================================================

DROP_COLUMNS = [
    "id",
    "attack_cat",
    "label",
]


CATEGORICAL_COLUMNS = [
    "proto",
    "service",
    "state",
]


SKEWED_COLUMNS = [
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "sloss",
    "dloss",
    "dinpkt",
    "sjit",
    "djit",
    "trans_depth",
    "response_body_len",
    "ct_flw_http_mthd",
]


# ============================================================
# HELPERS
# ============================================================

def create_preprocessor(numerical_columns):
    """
    Same preprocessing structure as the UNSW baseline:
        Numerical -> StandardScaler
        Categorical -> OneHotEncoder
    """

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical_columns,
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_COLUMNS,
            ),
        ]
    )


def apply_log_transform(df):
    """
    Apply log1p to the same 12 skewed columns
    used in the UNSW baseline experiment.
    """

    df = df.copy()

    for column in SKEWED_COLUMNS:
        df[column] = np.log1p(df[column])

    return df


def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    actual_fpr = fp / (fp + tn)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    return {
        "actual_fpr": float(actual_fpr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# LOAD DATA
# ============================================================

print("\n" + "=" * 70)
print("UNSW-NB15 HYBRID EXPERIMENT")
print("=" * 70)

print("\nLoading UNSW-NB15...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows:  {len(test_df):,}")


# ============================================================
# PREPARE X / y
# ============================================================

X_train_full = train_df.drop(columns=DROP_COLUMNS)
y_train_full = train_df["label"].astype(int)

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df["label"].astype(int)

FEATURE_COLUMNS = list(X_train_full.columns)

NUMERICAL_COLUMNS = [
    column
    for column in FEATURE_COLUMNS
    if column not in CATEGORICAL_COLUMNS
]

print(f"\nInput features: {len(FEATURE_COLUMNS)}")
print(f"Numerical features: {len(NUMERICAL_COLUMNS)}")
print(f"Categorical features: {len(CATEGORICAL_COLUMNS)}")


# ============================================================
# NORMAL-ONLY IF TRAINING
# ============================================================

print("\n" + "=" * 70)
print("ISOLATION FOREST")
print("=" * 70)

normal_mask = y_train_full == 0

X_normal = X_train_full.loc[normal_mask].copy()

print(
    f"\nNormal training records: "
    f"{len(X_normal):,}"
)


# EXACT SAME SPLIT AS BASELINE
X_fit, X_calib = train_test_split(
    X_normal,
    test_size=0.25,
    random_state=RANDOM_STATE,
)

print(
    f"Model-fit normal records: "
    f"{len(X_fit):,}"
)

print(
    f"Calibration normal records: "
    f"{len(X_calib):,}"
)


# ============================================================
# LOG TRANSFORMATION
# ============================================================

print("\nApplying log1p transformation...")

X_fit_if = apply_log_transform(X_fit)
X_calib_if = apply_log_transform(X_calib)
X_test_if = apply_log_transform(X_test)


# ============================================================
# IF PREPROCESSING
# ============================================================

print("\nFitting IF preprocessor...")

if_preprocessor = create_preprocessor(
    NUMERICAL_COLUMNS
)

X_fit_if_processed = (
    if_preprocessor.fit_transform(X_fit_if)
)

X_calib_if_processed = (
    if_preprocessor.transform(X_calib_if)
)

X_test_if_processed = (
    if_preprocessor.transform(X_test_if)
)

print(
    f"IF processed dimensions: "
    f"{X_fit_if_processed.shape}"
)


# ============================================================
# TRAIN ISOLATION FOREST
# ============================================================

print("\nTraining Isolation Forest...")

start_time = time.time()

if_model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    max_samples=MAX_SAMPLES,
    max_features=MAX_FEATURES,
    contamination=CONTAMINATION,
    bootstrap=False,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

if_model.fit(X_fit_if_processed)

training_time = time.time() - start_time

print(
    f"Training completed in "
    f"{training_time:.1f} seconds"
)


# ============================================================
# IF CALIBRATION
# ============================================================

print("\nCalibrating IF threshold...")

calibration_scores = (
    if_model.decision_function(
        X_calib_if_processed
    )
)


# Lower Isolation Forest score = more anomalous.
#
# Therefore:
#
# 5th percentile
#       ↓
# approximately 5% of normal calibration
# samples fall below this value.
#
# Those become anomalies.

if_threshold = float(
    np.quantile(
        calibration_scores,
        TARGET_FPR,
    )
)

print(
    f"Target calibration FPR: "
    f"{TARGET_FPR * 100:.2f}%"
)

print(
    f"IF threshold: "
    f"{if_threshold:.6f}"
)


# ============================================================
# IF TEST PREDICTIONS
# ============================================================

print("\nGenerating IF predictions...")

if_test_scores = (
    if_model.decision_function(
        X_test_if_processed
    )
)

if_predictions = (
    if_test_scores < if_threshold
).astype(int)


# ============================================================
# LOAD RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST")
print("=" * 70)

if not RF_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nRF model not found:\n{RF_MODEL_PATH}"
    )

if not RF_PREPROCESSOR_PATH.exists():
    raise FileNotFoundError(
        f"\nRF preprocessor not found:\n"
        f"{RF_PREPROCESSOR_PATH}"
    )


print("\nLoading saved RF model...")

rf_model = joblib.load(
    RF_MODEL_PATH
)

rf_preprocessor = joblib.load(
    RF_PREPROCESSOR_PATH
)


# ============================================================
# RF TEST PREDICTIONS
# ============================================================

print("Processing test data for RF...")

X_test_rf_processed = (
    rf_preprocessor.transform(X_test)
)

print(
    f"RF processed dimensions: "
    f"{X_test_rf_processed.shape}"
)


print(
    f"RF threshold: "
    f"{RF_THRESHOLD:.2f}"
)

rf_probabilities = (
    rf_model.predict_proba(
        X_test_rf_processed
    )[:, 1]
)


rf_predictions = (
    rf_probabilities >= RF_THRESHOLD
).astype(int)


# ============================================================
# HYBRID LOGIC
# ============================================================

print("\n" + "=" * 70)
print("HYBRID DECISION LOGIC")
print("=" * 70)


# ------------------------------------------------------------
# AND
# ------------------------------------------------------------
#
# Alert only if:
#
#     RF = attack
#     AND
#     IF = anomaly
#
# This should generally be more conservative.
#

hybrid_and_predictions = (
    (rf_predictions == 1)
    &
    (if_predictions == 1)
).astype(int)


# ------------------------------------------------------------
# OR
# ------------------------------------------------------------
#
# Alert if:
#
#     RF = attack
#     OR
#     IF = anomaly
#
# This should generally be more sensitive.
#

hybrid_or_predictions = (
    (rf_predictions == 1)
    |
    (if_predictions == 1)
).astype(int)


# ============================================================
# CALCULATE METRICS
# ============================================================

print("\nEvaluating models...")

if_metrics = calculate_metrics(
    y_test,
    if_predictions,
)

rf_metrics = calculate_metrics(
    y_test,
    rf_predictions,
)

and_metrics = calculate_metrics(
    y_test,
    hybrid_and_predictions,
)

or_metrics = calculate_metrics(
    y_test,
    hybrid_or_predictions,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 90)
print("FINAL HYBRID RESULTS")
print("=" * 90)

print(
    f"{'Model':<18}"
    f"{'FPR':>12}"
    f"{'Precision':>14}"
    f"{'Recall':>12}"
    f"{'F1':>12}"
)

print("-" * 90)

result_table = [
    ("Isolation Forest", if_metrics),
    ("Random Forest", rf_metrics),
    ("Hybrid AND", and_metrics),
    ("Hybrid OR", or_metrics),
]

for name, metrics in result_table:

    print(
        f"{name:<18}"
        f"{metrics['actual_fpr'] * 100:>11.2f}%"
        f"{metrics['precision'] * 100:>13.2f}%"
        f"{metrics['recall'] * 100:>11.2f}%"
        f"{metrics['f1']:>12.4f}"
    )


# ============================================================
# CONFUSION MATRICES
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRICES")
print("=" * 70)


for name, metrics in result_table:

    print(f"\n{name}")

    print(
        f"TN = {metrics['tn']}"
    )

    print(
        f"FP = {metrics['fp']}"
    )

    print(
        f"FN = {metrics['fn']}"
    )

    print(
        f"TP = {metrics['tp']}"
    )


# ============================================================
# SAVE REPORT
# ============================================================

OUTPUT_PATH = (
    REPORTS_DIR
    / "unsw_nb15_hybrid_comparison.json"
)


report = {
    "experiment": "UNSW-NB15 IF + RF hybrid",
    "dataset": "UNSW-NB15",

    "training_file": TRAIN_PATH.name,
    "testing_file": TEST_PATH.name,

    "methodology": {
        "if_normal_only_training": True,
        "if_normal_split": (
            "75% model fit / 25% calibration"
        ),
        "random_state": RANDOM_STATE,
        "if_target_fpr": TARGET_FPR,

        "rf_threshold": RF_THRESHOLD,

        "official_test_used_only_for_evaluation": True,

        "threshold_tuning_on_official_test": False,
    },

    "isolation_forest": {
        "algorithm": "IsolationForest",

        "n_estimators": N_ESTIMATORS,
        "max_samples": MAX_SAMPLES,
        "max_features": MAX_FEATURES,
        "contamination": CONTAMINATION,

        "input_features": len(FEATURE_COLUMNS),

        "numerical_features": len(
            NUMERICAL_COLUMNS
        ),

        "categorical_features": len(
            CATEGORICAL_COLUMNS
        ),

        "processed_dimensions": int(
            X_fit_if_processed.shape[1]
        ),

        "normal_fit_rows": len(X_fit),

        "normal_calibration_rows": len(
            X_calib
        ),

        "target_fpr": TARGET_FPR,

        "threshold": if_threshold,

        "metrics": if_metrics,
    },

    "random_forest": {
        "model_path": str(
            RF_MODEL_PATH
        ),

        "preprocessor_path": str(
            RF_PREPROCESSOR_PATH
        ),

        "threshold": RF_THRESHOLD,

        "processed_dimensions": int(
            X_test_rf_processed.shape[1]
        ),

        "metrics": rf_metrics,
    },

    "hybrid_and": {
        "logic": (
            "RF predicts attack AND "
            "IF predicts anomaly"
        ),

        "metrics": and_metrics,
    },

    "hybrid_or": {
        "logic": (
            "RF predicts attack OR "
            "IF predicts anomaly"
        ),

        "metrics": or_metrics,
    },

    "comparison": {
        "isolation_forest": if_metrics,
        "random_forest": rf_metrics,
        "hybrid_and": and_metrics,
        "hybrid_or": or_metrics,
    },
}


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )


# ============================================================
# DONE
# ============================================================

print("\n" + "=" * 90)
print("HYBRID EXPERIMENT COMPLETE")
print("=" * 90)

print(
    f"\nReport saved to:\n{OUTPUT_PATH}"
)

print("\nThresholds used:")
print(
    f"IF threshold = {if_threshold:.6f}"
)
print(
    f"RF threshold = {RF_THRESHOLD:.2f}"
)

print(
    "\nNo threshold tuning was performed "
    "on the official test set."
)