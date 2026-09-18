import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "UNSW_NB15_training-set.csv"
TEST_PATH = ROOT / "data" / "UNSW_NB15_testing-set.csv"
REPORT_PATH = ROOT / "reports" / "unsw_nb15_threshold_tuning.json"

RANDOM_STATE = 42

DROP_COLUMNS = ["id", "attack_cat", "label"]

LOG_COLUMNS = [
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

TARGET_FPRS = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.06,
    0.07,
    0.08,
    0.09,
    0.10,
]


def apply_log1p(df):
    df = df.copy()

    for col in LOG_COLUMNS:
        df[col] = np.log1p(df[col])

    return df


def build_preprocessor(X):
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numerical = [c for c in X.columns if c not in categorical]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical,
            ),
        ]
    )


print("Loading UNSW-NB15...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=DROP_COLUMNS)
y_train = train_df["label"].astype(int)

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df["label"].astype(int)

# Only normal traffic is used for model fitting/calibration
X_normal = X_train[y_train == 0]

X_fit, X_calib = train_test_split(
    X_normal,
    test_size=0.25,
    random_state=RANDOM_STATE,
)

print(f"Normal training records: {len(X_normal):,}")
print(f"Model-fit records:       {len(X_fit):,}")
print(f"Calibration records:     {len(X_calib):,}")


# ---------------------------------------------------------
# FIXED PREPROCESSING
# ---------------------------------------------------------

print("\nApplying Log1p transformation...")

X_fit = apply_log1p(X_fit)
X_calib = apply_log1p(X_calib)
X_test = apply_log1p(X_test)

print("Fitting preprocessor...")

preprocessor = build_preprocessor(X_fit)

X_fit_processed = preprocessor.fit_transform(X_fit)
X_calib_processed = preprocessor.transform(X_calib)
X_test_processed = preprocessor.transform(X_test)

print(
    f"Processed dimensions: {X_fit_processed.shape}"
)


# ---------------------------------------------------------
# FIXED ISOLATION FOREST
# ---------------------------------------------------------

print("\nTraining fixed Isolation Forest...")

model = IsolationForest(
    n_estimators=800,
    max_samples=1.0,
    max_features=0.7,
    contamination=0.01,
    bootstrap=False,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

start = time.time()

model.fit(X_fit_processed)

print(
    f"Training time: {time.time() - start:.1f} seconds"
)


# ---------------------------------------------------------
# GET SCORES
# ---------------------------------------------------------

print("\nCalculating calibration scores...")

calib_scores = model.decision_function(
    X_calib_processed
)

print("Calculating test scores...")

test_scores = model.decision_function(
    X_test_processed
)

# Lower Isolation Forest score = more anomalous.
# Therefore, attack-like score is -decision_function().
attack_scores = -test_scores

# AUC metrics do not depend on the chosen threshold.
roc_auc = roc_auc_score(
    y_test,
    attack_scores,
)

pr_auc = average_precision_score(
    y_test,
    attack_scores,
)


# ---------------------------------------------------------
# THRESHOLD SWEEP
# ---------------------------------------------------------

results = []

for target_fpr in TARGET_FPRS:

    print("\n" + "=" * 70)
    print(
        f"TARGET FPR: {target_fpr:.0%}"
    )
    print("=" * 70)

    # Because lower scores are more anomalous,
    # the target FPR corresponds to this lower quantile.
    threshold = float(
        np.quantile(
            calib_scores,
            target_fpr,
        )
    )

    # Score below threshold => predicted attack
    y_pred = (
        test_scores < threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
    ).ravel()

    actual_fpr = fp / (fp + tn)

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
        "target_fpr": target_fpr,
        "threshold": threshold,
        "actual_test_fpr": float(actual_fpr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }

    results.append(result)

    print(
        f"Threshold:       {threshold:.6f}"
    )
    print(
        f"Actual Test FPR: {actual_fpr:.2%}"
    )
    print(
        f"Precision:       {precision:.4f}"
    )
    print(
        f"Recall:          {recall:.4f}"
    )
    print(
        f"F1:              {f1:.4f}"
    )
    print(
        f"ROC-AUC:         {roc_auc:.4f}"
    )
    print(
        f"PR-AUC:          {pr_auc:.4f}"
    )


# ---------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------

report = {
    "dataset": "UNSW-NB15",
    "experiment": "threshold tuning",
    "preprocessing": (
        "log1p + StandardScaler + OneHotEncoder"
    ),
    "model": {
        "algorithm": "IsolationForest",
        "n_estimators": 800,
        "max_samples": 1.0,
        "max_features": 0.7,
        "contamination": 0.01,
        "random_state": RANDOM_STATE,
    },
    "calibration": {
        "normal_fit_fraction": 0.75,
        "normal_calibration_fraction": 0.25,
        "random_state": RANDOM_STATE,
    },
    "results": results,
}

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(REPORT_PATH, "w") as f:
    json.dump(
        report,
        f,
        indent=2,
    )


# ---------------------------------------------------------
# FINAL TABLE
# ---------------------------------------------------------

print("\n\n" + "=" * 80)
print("FINAL THRESHOLD COMPARISON")
print("=" * 80)

print(
    f"{'Target':>8}"
    f"{'Actual FPR':>12}"
    f"{'Precision':>12}"
    f"{'Recall':>10}"
    f"{'F1':>10}"
    f"{'Threshold':>14}"
)

for r in results:

    print(
        f"{r['target_fpr']:>7.0%}"
        f"{r['actual_test_fpr']:>11.2%}"
        f"{r['precision']:>12.4f}"
        f"{r['recall']:>10.4f}"
        f"{r['f1']:>10.4f}"
        f"{r['threshold']:>14.6f}"
    )

print("\nReport created:")
print(REPORT_PATH)