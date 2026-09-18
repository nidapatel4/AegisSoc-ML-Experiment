import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "data" / "UNSW_NB15_training-set.csv"
TEST_PATH = BASE_DIR / "data" / "UNSW_NB15_testing-set.csv"


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


def apply_log1p(df):
    df = df.copy()

    for col in SKEWED_COLUMNS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))

    return df


print("Loading UNSW-NB15...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

y_train = train_df["label"].values
y_test = test_df["label"].values

attack_categories = test_df["attack_cat"].values

DROP_COLUMNS = ["id", "attack_cat", "label"]

X_train = train_df.drop(columns=DROP_COLUMNS)
X_test = test_df.drop(columns=DROP_COLUMNS)

# ------------------------------------------------------------
# NORMAL TRAINING DATA
# ------------------------------------------------------------

normal_mask = y_train == 0

X_normal = X_train[normal_mask]

print(f"Normal training records: {len(X_normal):,}")

X_fit = X_normal.sample(frac=0.75, random_state=42)
X_calib = X_normal.drop(X_fit.index)

print(f"Model-fit records:       {len(X_fit):,}")
print(f"Calibration records:     {len(X_calib):,}")

# ------------------------------------------------------------
# PREPROCESSING
# ------------------------------------------------------------

print("\nApplying Log1p...")

X_fit = apply_log1p(X_fit)
X_calib = apply_log1p(X_calib)
X_test = apply_log1p(X_test)

categorical_features = [
    "proto",
    "service",
    "state",
]

numeric_features = [
    col
    for col in X_fit.columns
    if col not in categorical_features
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features,
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
            categorical_features,
        ),
    ]
)

print("Fitting preprocessor...")

X_fit_processed = preprocessor.fit_transform(X_fit)
X_calib_processed = preprocessor.transform(X_calib)
X_test_processed = preprocessor.transform(X_test)

print(
    f"Processed dimensions: {X_fit_processed.shape[1]}"
)

# ------------------------------------------------------------
# ISOLATION FOREST
# ------------------------------------------------------------

print("\nTraining Isolation Forest...")

model = IsolationForest(
    n_estimators=800,
    max_samples=1.0,
    max_features=0.7,
    contamination=0.01,
    bootstrap=False,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_fit_processed)

# ------------------------------------------------------------
# THRESHOLD CALIBRATION
# ------------------------------------------------------------

print("Calibrating threshold at 5% target FPR...")

calib_scores = model.decision_function(
    X_calib_processed
)

threshold = np.quantile(
    calib_scores,
    0.05,
)

print(f"Threshold: {threshold:.6f}")

# ------------------------------------------------------------
# TEST PREDICTIONS
# ------------------------------------------------------------

test_scores = model.decision_function(
    X_test_processed
)

y_pred = (
    test_scores < threshold
).astype(int)

# ------------------------------------------------------------
# OVERALL RESULTS
# ------------------------------------------------------------

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1],
).ravel()

overall_fpr = fp / (fp + tn)
overall_recall = tp / (tp + fn)

print("\n" + "=" * 75)
print("OVERALL RESULTS")
print("=" * 75)

print(f"FPR:    {overall_fpr:.2%}")
print(f"Recall: {overall_recall:.2%}")

# ------------------------------------------------------------
# ATTACK FAMILY ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("ATTACK FAMILY ANALYSIS")
print("=" * 75)

results = []

attack_df = test_df[test_df["label"] == 1].copy()

for category in sorted(
    attack_df["attack_cat"].dropna().unique()
):

    mask = (
        (test_df["label"].values == 1)
        & (attack_categories == category)
    )

    total = int(mask.sum())

    detected = int(
        (y_pred[mask] == 1).sum()
    )

    missed = total - detected

    recall = (
        detected / total
        if total > 0
        else 0
    )

    results.append(
        {
            "attack_category": category,
            "total": total,
            "detected": detected,
            "missed": missed,
            "recall": recall,
        }
    )

    print(
        f"{category:<15}"
        f" Total: {total:>6,}"
        f"  Detected: {detected:>6,}"
        f"  Missed: {missed:>6,}"
        f"  Recall: {recall:>7.2%}"
    )

# ------------------------------------------------------------
# SAVE REPORT
# ------------------------------------------------------------

report = {
    "model": "IsolationForest",
    "preprocessing": (
        "Log1p + StandardScaler + OneHotEncoder"
    ),
    "target_fpr": 0.05,
    "threshold": float(threshold),
    "overall_fpr": float(overall_fpr),
    "overall_recall": float(overall_recall),
    "attack_families": results,
}

REPORT_PATH = (
    BASE_DIR
    / "reports"
    / "unsw_nb15_attack_family_analysis.json"
)

with open(REPORT_PATH, "w") as f:
    json.dump(
        report,
        f,
        indent=2,
    )

print("\nReport created:")
print(REPORT_PATH)