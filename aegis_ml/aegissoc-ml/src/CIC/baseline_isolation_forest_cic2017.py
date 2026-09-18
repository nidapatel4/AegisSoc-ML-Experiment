import json
from pathlib import Path

import joblib
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "models" / "CIC"
REPORT_DIR = BASE_DIR / "reports" / "CIC"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TARGET_FPR = 0.05

CONTAMINATION = 0.01

N_ESTIMATORS = 800

MAX_SAMPLES = 1.0

MAX_FEATURES = 0.7


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CIC-IDS2017 BASELINE ISOLATION FOREST")
print("=" * 70)

print("\nLoading preprocessed data...")

X_train = np.load(
    MODEL_DIR / "X_train.npy"
)

X_calibration = np.load(
    MODEL_DIR / "X_calibration.npy"
)

X_test = np.load(
    MODEL_DIR / "X_test.npy"
)

y_test = np.load(
    MODEL_DIR / "y_test.npy"
)

print(f"Training shape:      {X_train.shape}")
print(f"Calibration shape:   {X_calibration.shape}")
print(f"Test shape:         {X_test.shape}")

print(f"Test normal:        {(y_test == 0).sum():,}")
print(f"Test attacks:       {(y_test == 1).sum():,}")


# ============================================================
# TRAIN BASELINE ISOLATION FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING BASELINE")
print("=" * 70)

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    max_samples=MAX_SAMPLES,
    max_features=MAX_FEATURES,
    contamination=CONTAMINATION,
    bootstrap=False,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("Training Isolation Forest...")

model.fit(X_train)

print("Training complete.")


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR /
    "cic2017_isolation_forest_baseline.joblib"
)

joblib.dump(
    model,
    model_path
)

print(f"\nModel saved:")
print(model_path)


# ============================================================
# GET ANOMALY SCORES
# ============================================================
#
# Isolation Forest's decision_function:
#
#   higher score → more normal
#   lower score  → more anomalous
#
# Therefore, LOWER scores should be classified as attacks.
# ============================================================

print("\nGenerating scores...")

calibration_scores = model.decision_function(
    X_calibration
)

test_scores = model.decision_function(
    X_test
)


# ============================================================
# CALIBRATE THRESHOLD
# ============================================================
#
# Calibration contains NORMAL traffic only.
#
# Target FPR = 5%
#
# Therefore, choose the score threshold such that
# approximately 5% of calibration normal traffic falls
# below the threshold.
# ============================================================

threshold = np.percentile(
    calibration_scores,
    TARGET_FPR * 100
)

print("\n" + "=" * 70)
print("THRESHOLD CALIBRATION")
print("=" * 70)

print(f"Target FPR:          {TARGET_FPR:.2%}")
print(f"Threshold:            {threshold:.6f}")


# ============================================================
# CLASSIFY TEST DATA
# ============================================================
#
# Lower score than threshold → attack
# Score >= threshold         → normal
# ============================================================

y_pred = (
    test_scores < threshold
).astype(int)


# ============================================================
# METRICS
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
).ravel()

fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    -test_scores
)

pr_auc = average_precision_score(
    y_test,
    -test_scores
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("BASELINE TEST RESULTS")
print("=" * 70)

print(f"Accuracy:             {accuracy:.4f}")
print(f"Precision:            {precision:.4f}")
print(f"Recall:               {recall:.4f}")
print(f"F1:                   {f1:.4f}")
print(f"ROC-AUC:              {roc_auc:.4f}")
print(f"PR-AUC:               {pr_auc:.4f}")
print(f"Actual test FPR:      {fpr:.4f}")

print("\nConfusion Matrix:")
print(
    f"TN = {tn:,}"
)
print(
    f"FP = {fp:,}"
)
print(
    f"FN = {fn:,}"
)
print(
    f"TP = {tp:,}"
)


# ============================================================
# SAVE REPORT
# ============================================================

report = {
    "dataset": "CIC-IDS2017",

    "experiment": "baseline_isolation_forest",

    "model": {
        "algorithm": "IsolationForest",
        "n_estimators": N_ESTIMATORS,
        "max_samples": MAX_SAMPLES,
        "max_features": MAX_FEATURES,
        "contamination": CONTAMINATION,
        "bootstrap": False,
        "random_state": RANDOM_STATE,
    },

    "data": {
        "training_rows": int(len(X_train)),
        "calibration_rows": int(len(X_calibration)),
        "test_rows": int(len(X_test)),
        "test_normal": int((y_test == 0).sum()),
        "test_attacks": int((y_test == 1).sum()),
        "features": int(X_train.shape[1]),
    },

    "threshold_calibration": {
        "target_fpr": TARGET_FPR,
        "threshold": float(threshold),
        "calibration_rows": int(len(X_calibration)),
    },

    "metrics": {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "actual_test_fpr": float(fpr),
    },

    "confusion_matrix": {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    },

    "notes": [
        "Model trained using normal training traffic only.",
        "Threshold calibrated using normal calibration traffic only.",
        "Test set was not used during model fitting or threshold selection.",
        "Lower Isolation Forest decision_function scores indicate more anomalous traffic.",
    ],
}

report_path = (
    REPORT_DIR /
    "cic2017_baseline_isolation_forest.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


# ============================================================
# COMPLETE
# ============================================================

print("\nReport saved:")
print(report_path)

print("\n" + "=" * 70)
print("BASELINE EXPERIMENT COMPLETE")
print("=" * 70)