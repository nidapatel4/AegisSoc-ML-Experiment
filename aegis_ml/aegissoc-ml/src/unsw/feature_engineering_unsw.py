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
REPORT_PATH = ROOT / "reports" / "unsw_nb15_feature_engineering.json"

RANDOM_STATE = 42
TARGET_FPR = 0.05

DROP_COLUMNS = ["id", "attack_cat", "label"]

LOG_COLUMNS = [
    "spkts", "dpkts", "sbytes", "dbytes",
    "sloss", "dloss", "dinpkt", "sjit",
    "djit", "trans_depth",
    "response_body_len", "ct_flw_http_mthd",
]


def safe_divide(a, b):
    return a / (b.replace(0, np.nan) + 1e-9)


def engineer_features(df):
    df = df.copy()

    # Security-oriented derived network features
    df["bytes_per_packet"] = (
        (df["sbytes"] + df["dbytes"]) /
        (df["spkts"] + df["dpkts"] + 1)
    )

    df["src_dst_byte_ratio"] = safe_divide(
        df["sbytes"],
        df["dbytes"] + 1
    )

    df["packet_ratio"] = safe_divide(
        df["spkts"],
        df["dpkts"] + 1
    )

    df["loss_rate"] = (
        (df["sloss"] + df["dloss"]) /
        (df["spkts"] + df["dpkts"] + 1)
    )

    df["connection_rate"] = (
        df["ct_srv_src"] +
        df["ct_dst_ltm"] +
        df["ct_src_ltm"]
    )

    return df


def apply_log1p(df):
    df = df.copy()

    for col in LOG_COLUMNS:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    return df


def build_preprocessor(X):
    categorical = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical = [
        c for c in X.columns
        if c not in categorical
    ]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numerical,
            ),
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


def evaluate(X_fit, X_calib, X_test, y_test):

    preprocessor = build_preprocessor(X_fit)

    X_fit = preprocessor.fit_transform(X_fit)
    X_calib = preprocessor.transform(X_calib)
    X_test = preprocessor.transform(X_test)

    print(
        f"Processed dimensions: {X_fit.shape[1]}"
    )

    model = IsolationForest(
        n_estimators=800,
        max_samples=1.0,
        max_features=0.7,
        contamination=0.01,
        bootstrap=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_fit)

    calib_scores = model.decision_function(
        X_calib
    )

    threshold = float(
        np.quantile(
            calib_scores,
            TARGET_FPR,
        )
    )

    test_scores = model.decision_function(
        X_test
    )

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

    attack_scores = -test_scores

    roc_auc = roc_auc_score(
        y_test,
        attack_scores,
    )

    pr_auc = average_precision_score(
        y_test,
        attack_scores,
    )

    return {
        "processed_dimensions": X_fit.shape[1],
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


print("Loading UNSW-NB15...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

y_train = train_df["label"].astype(int)
y_test = test_df["label"].astype(int)

X_train = train_df.drop(
    columns=DROP_COLUMNS
)

X_test = test_df.drop(
    columns=DROP_COLUMNS
)

# Only normal records enter model fitting/calibration
X_normal = X_train[
    y_train == 0
]

X_fit, X_calib = train_test_split(
    X_normal,
    test_size=0.25,
    random_state=RANDOM_STATE,
)

print(
    f"Normal training records: {len(X_normal):,}"
)
print(
    f"Model-fit records:       {len(X_fit):,}"
)
print(
    f"Calibration records:     {len(X_calib):,}"
)


# =========================================================
# EXPERIMENTS
# =========================================================

feature_sets = {

    "baseline": [],

    "all_engineered": [
        "bytes_per_packet",
        "src_dst_byte_ratio",
        "packet_ratio",
        "loss_rate",
        "connection_rate",
    ],

    "without_bytes_per_packet": [
        "src_dst_byte_ratio",
        "packet_ratio",
        "loss_rate",
        "connection_rate",
    ],

    "without_src_dst_byte_ratio": [
        "bytes_per_packet",
        "packet_ratio",
        "loss_rate",
        "connection_rate",
    ],

    "without_packet_ratio": [
        "bytes_per_packet",
        "src_dst_byte_ratio",
        "loss_rate",
        "connection_rate",
    ],

    "without_loss_rate": [
        "bytes_per_packet",
        "src_dst_byte_ratio",
        "packet_ratio",
        "connection_rate",
    ],

    "without_connection_rate": [
        "bytes_per_packet",
        "src_dst_byte_ratio",
        "packet_ratio",
        "loss_rate",
    ],
}


results = []


for name, engineered_columns in feature_sets.items():

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)

    start = time.time()

    fit_df = X_fit.copy()
    calib_df = X_calib.copy()
    test_df_processed = X_test.copy()

    if engineered_columns:

        fit_df = engineer_features(fit_df)
        calib_df = engineer_features(calib_df)
        test_df_processed = engineer_features(
            test_df_processed
        )

    # Keep only selected engineered features
    # while preserving original features
    if not engineered_columns:

        pass

    print("Applying Log1p...")

    fit_df = apply_log1p(fit_df)
    calib_df = apply_log1p(calib_df)
    test_df_processed = apply_log1p(
        test_df_processed
    )

    # Remove engineered features not belonging
    # to this experiment
    all_engineered = [
        "bytes_per_packet",
        "src_dst_byte_ratio",
        "packet_ratio",
        "loss_rate",
        "connection_rate",
    ]

    for col in all_engineered:

        if col not in engineered_columns:

            if col in fit_df.columns:
                fit_df = fit_df.drop(
                    columns=[col]
                )

            if col in calib_df.columns:
                calib_df = calib_df.drop(
                    columns=[col]
                )

            if col in test_df_processed.columns:
                test_df_processed = (
                    test_df_processed.drop(
                        columns=[col]
                    )
                )

    result = evaluate(
        fit_df,
        calib_df,
        test_df_processed,
        y_test,
    )

    result["name"] = name
    result["engineered_features"] = (
        engineered_columns
    )
    result["time_seconds"] = round(
        time.time() - start,
        1,
    )

    results.append(result)

    print("\nRESULTS")
    print("-" * 50)
    print(
        f"FPR:       {result['actual_test_fpr']:.2%}"
    )
    print(
        f"Precision: {result['precision']:.4f}"
    )
    print(
        f"Recall:    {result['recall']:.4f}"
    )
    print(
        f"F1:        {result['f1']:.4f}"
    )
    print(
        f"ROC-AUC:   {result['roc_auc']:.4f}"
    )
    print(
        f"PR-AUC:    {result['pr_auc']:.4f}"
    )


# =========================================================
# SAVE REPORT
# =========================================================

report = {
    "dataset": "UNSW-NB15",
    "experiment": "feature engineering and ablation",
    "preprocessing": (
        "Log1p + StandardScaler + OneHotEncoder"
    ),
    "model": {
        "algorithm": "IsolationForest",
        "n_estimators": 800,
        "max_samples": 1.0,
        "max_features": 0.7,
        "contamination": 0.01,
        "random_state": RANDOM_STATE,
    },
    "target_fpr": TARGET_FPR,
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


# =========================================================
# FINAL TABLE
# =========================================================

print("\n\n" + "=" * 90)
print("FINAL FEATURE ENGINEERING COMPARISON")
print("=" * 90)

print(
    f"{'Experiment':30}"
    f"{'FPR':>9}"
    f"{'Precision':>12}"
    f"{'Recall':>10}"
    f"{'F1':>10}"
    f"{'ROC-AUC':>11}"
    f"{'PR-AUC':>10}"
)

for r in results:

    print(
        f"{r['name']:30}"
        f"{r['actual_test_fpr']:>8.2%}"
        f"{r['precision']:>12.4f}"
        f"{r['recall']:>10.4f}"
        f"{r['f1']:>10.4f}"
        f"{r['roc_auc']:>11.4f}"
        f"{r['pr_auc']:>10.4f}"
    )

print("\nReport created:")
print(REPORT_PATH)