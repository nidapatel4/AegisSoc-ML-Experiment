from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = ROOT / "data" / "UNSW_NB15_testing-set.csv"

MODEL_PATH = ROOT / "models" / "unsw_random_forest.joblib"
PREPROCESSOR_PATH = (
    ROOT / "models" / "unsw_random_forest_preprocessor.joblib"
)

REPORT_PATH = (
    ROOT
    / "reports"
    / "unsw_nb15_random_forest_threshold_sweep.json"
)


# ============================================================
# THRESHOLDS
# ============================================================

THRESHOLDS = [
    0.10,
    0.20,
    0.30,
    0.40,
    0.50,
    0.55,
    0.60,
    0.63,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]


# ============================================================
# LOAD MODEL + PREPROCESSOR
# ============================================================

print("\nLoading saved Random Forest...")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

print("Model loaded.")
print("Preprocessor loaded.")


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading UNSW-NB15 testing set...")

test_df = pd.read_csv(TEST_PATH)

DROP_COLUMNS = [
    "id",
    "attack_cat",
    "label",
]

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df["label"].astype(int)


# ============================================================
# APPLY SAME LOG1P TRANSFORMATION
# ============================================================

LOG_FEATURES = [
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


def apply_log1p(df):
    df = df.copy()

    for column in LOG_FEATURES:
        if column in df.columns:
            df[column] = np.log1p(
                df[column].clip(lower=0)
            )

    return df


X_test = apply_log1p(X_test)


# ============================================================
# PREPROCESS
# ============================================================

print("\nTransforming test data...")

X_test_processed = preprocessor.transform(X_test)

print(
    f"Processed test shape: "
    f"{X_test_processed.shape}"
)


# ============================================================
# GENERATE PROBABILITIES ONCE
# ============================================================

print("\nGenerating attack probabilities...")

test_probabilities = model.predict_proba(
    X_test_processed
)[:, 1]

print("Probabilities generated.")


# ============================================================
# THRESHOLD SWEEP
# ============================================================

print("\nRunning threshold sweep...\n")

results = []

for threshold in THRESHOLDS:

    # Probability >= threshold → attack
    y_pred = (
        test_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    ).ravel()

    actual_fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    result = {
        "threshold": threshold,
        "actual_fpr": actual_fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    results.append(result)

    print(
        f"Threshold {threshold:.2f} | "
        f"FPR {actual_fpr:.2%} | "
        f"Precision {precision:.4f} | "
        f"Recall {recall:.4f} | "
        f"F1 {f1:.4f}"
    )


# ============================================================
# BEST F1
# ============================================================

best_f1 = max(
    results,
    key=lambda x: x["f1"],
)


# ============================================================
# CLOSEST TO 5% FPR
# ============================================================

closest_to_5_fpr = min(
    results,
    key=lambda x: abs(
        x["actual_fpr"] - 0.05
    ),
)


# ============================================================
# SAVE REPORT
# ============================================================

report = {
    "dataset": "UNSW-NB15",
    "model": "RandomForestClassifier",
    "model_path": str(MODEL_PATH),
    "preprocessor_path": str(PREPROCESSOR_PATH),

    "thresholds_tested": THRESHOLDS,

    "results": results,

    "best_f1": best_f1,

    "closest_to_5_percent_actual_fpr": closest_to_5_fpr,
}


with open(REPORT_PATH, "w") as f:
    json.dump(
        report,
        f,
        indent=2,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST THRESHOLD SWEEP COMPLETE")
print("=" * 70)

print("\nBest F1:")
print(
    f"Threshold: {best_f1['threshold']:.2f}"
)
print(
    f"FPR:       {best_f1['actual_fpr']:.2%}"
)
print(
    f"Precision: {best_f1['precision']:.4f}"
)
print(
    f"Recall:    {best_f1['recall']:.4f}"
)
print(
    f"F1:        {best_f1['f1']:.4f}"
)

print("\nClosest to 5% actual FPR:")
print(
    f"Threshold: "
    f"{closest_to_5_fpr['threshold']:.2f}"
)
print(
    f"FPR:       "
    f"{closest_to_5_fpr['actual_fpr']:.2%}"
)
print(
    f"Precision: "
    f"{closest_to_5_fpr['precision']:.4f}"
)
print(
    f"Recall:    "
    f"{closest_to_5_fpr['recall']:.4f}"
)
print(
    f"F1:        "
    f"{closest_to_5_fpr['f1']:.4f}"
)

print("\nReport saved to:")
print(REPORT_PATH)

print("\nDone.")