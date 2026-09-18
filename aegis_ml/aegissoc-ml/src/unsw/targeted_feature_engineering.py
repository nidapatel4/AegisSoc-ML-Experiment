import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "data" / "UNSW_NB15_training-set.csv"
TEST_PATH = BASE_DIR / "data" / "UNSW_NB15_testing-set.csv"


# These were identified from the feature-distribution analysis.
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
            df[col] = np.log1p(
                df[col].clip(lower=0)
            )

    return df


def add_targeted_features(df):
    """
    Add features motivated by the attack-family analysis.

    These capture:
    - traffic volume
    - packet volume
    - traffic rate
    - source/destination imbalance
    - packet loss
    - connection activity
    """

    df = df.copy()

    # --------------------------------------------------------
    # 1. Total traffic volume
    # --------------------------------------------------------

    df["total_bytes"] = (
        df["sbytes"] + df["dbytes"]
    )

    df["total_packets"] = (
        df["spkts"] + df["dpkts"]
    )

    # --------------------------------------------------------
    # 2. Source/destination imbalance
    # --------------------------------------------------------

    df["byte_ratio"] = (
        df["sbytes"] /
        (df["dbytes"] + 1)
    )

    df["packet_ratio"] = (
        df["spkts"] /
        (df["dpkts"] + 1)
    )

    # --------------------------------------------------------
    # 3. Traffic load imbalance
    # --------------------------------------------------------

    df["load_ratio"] = (
        df["sload"] /
        (df["dload"] + 1)
    )

    # --------------------------------------------------------
    # 4. Packet loss rate
    # --------------------------------------------------------

    df["loss_rate"] = (
        (df["sloss"] + df["dloss"]) /
        (df["spkts"] + df["dpkts"] + 1)
    )

    # --------------------------------------------------------
    # 5. Connection/activity intensity
    # --------------------------------------------------------

    df["connection_activity"] = (
        df["ct_srv_src"]
        + df["ct_srv_dst"]
        + df["ct_dst_ltm"]
        + df["ct_src_ltm"]
        + df["ct_dst_src_ltm"]
    )

    # --------------------------------------------------------
    # 6. TCP handshake activity
    # --------------------------------------------------------

    df["tcp_handshake_time"] = (
        df["synack"]
        + df["ackdat"]
        + df["tcprtt"]
    )

    return df


# ============================================================
# LOAD DATA
# ============================================================

print("Loading UNSW-NB15...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

y_train = train_df["label"].values
y_test = test_df["label"].values


DROP_COLUMNS = [
    "id",
    "attack_cat",
    "label",
]

X_train = train_df.drop(
    columns=DROP_COLUMNS
)

X_test = test_df.drop(
    columns=DROP_COLUMNS
)


# ============================================================
# NORMAL TRAINING DATA
# ============================================================

normal_mask = y_train == 0

X_normal = X_train[normal_mask]

print(
    f"Normal training records: {len(X_normal):,}"
)

X_fit = X_normal.sample(
    frac=0.75,
    random_state=42,
)

X_calib = X_normal.drop(
    X_fit.index
)

print(
    f"Model-fit records:       {len(X_fit):,}"
)

print(
    f"Calibration records:     {len(X_calib):,}"
)


# ============================================================
# ADD TARGETED FEATURES
# ============================================================

print("\nAdding targeted features...")

X_fit = add_targeted_features(X_fit)
X_calib = add_targeted_features(X_calib)
X_test = add_targeted_features(X_test)


# ============================================================
# PREPROCESSING
# ============================================================

print("Applying Log1p...")

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

X_fit_processed = preprocessor.fit_transform(
    X_fit
)

X_calib_processed = preprocessor.transform(
    X_calib
)

X_test_processed = preprocessor.transform(
    X_test
)

print(
    f"Processed dimensions: "
    f"{X_fit_processed.shape[1]}"
)


# ============================================================
# ISOLATION FOREST
# ============================================================

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


# ============================================================
# CALIBRATION
# ============================================================

print(
    "Calibrating threshold at 5% target FPR..."
)

calib_scores = model.decision_function(
    X_calib_processed
)

threshold = np.quantile(
    calib_scores,
    0.05,
)

print(
    f"Threshold: {threshold:.6f}"
)


# ============================================================
# TEST
# ============================================================

test_scores = model.decision_function(
    X_test_processed
)

y_pred = (
    test_scores < threshold
).astype(int)


# ============================================================
# METRICS
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1],
).ravel()

fpr = fp / (fp + tn)

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

roc_auc = roc_auc_score(
    y_test,
    -test_scores,
)

pr_auc = average_precision_score(
    y_test,
    -test_scores,
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 65)
print("TARGETED FEATURE ENGINEERING RESULTS")
print("=" * 65)

print(
    f"FPR:       {fpr:.2%}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1:        {f1:.4f}"
)

print(
    f"ROC-AUC:   {roc_auc:.4f}"
)

print(
    f"PR-AUC:    {pr_auc:.4f}"
)

print("\nConfusion Matrix")

print(
    f"TN: {tn:,}"
)

print(
    f"FP: {fp:,}"
)

print(
    f"FN: {fn:,}"
)

print(
    f"TP: {tp:,}"
)


# ============================================================
# SAVE REPORT
# ============================================================

results = {
    "experiment": "targeted_feature_engineering",
    "features_added": [
        "total_bytes",
        "total_packets",
        "byte_ratio",
        "packet_ratio",
        "load_ratio",
        "loss_rate",
        "connection_activity",
        "tcp_handshake_time",
    ],
    "preprocessing": (
        "Log1p + StandardScaler + OneHotEncoder"
    ),
    "model": "IsolationForest",
    "n_estimators": 800,
    "max_samples": 1.0,
    "max_features": 0.7,
    "target_fpr": 0.05,
    "threshold": float(threshold),
    "fpr": float(fpr),
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


REPORT_PATH = (
    BASE_DIR
    / "reports"
    / "unsw_nb15_targeted_feature_engineering.json"
)

with open(REPORT_PATH, "w") as f:
    json.dump(
        results,
        f,
        indent=2,
    )

print("\nReport created:")
print(REPORT_PATH)