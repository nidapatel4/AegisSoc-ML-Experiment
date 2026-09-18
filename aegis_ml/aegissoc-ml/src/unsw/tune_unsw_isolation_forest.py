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
REPORT_PATH = ROOT / "reports" / "unsw_nb15_hyperparameter_tuning.json"

TARGET_FPR = 0.05
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


def apply_log1p(df):
    df = df.copy()

    for col in LOG_COLUMNS:
        df[col] = np.log1p(df[col])

    return df


def build_preprocessor(X):
    categorical = X.select_dtypes(include=["object"]).columns.tolist()
    numerical = [c for c in X.columns if c not in categorical]

    preprocessor = ColumnTransformer(
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

    return preprocessor


def evaluate_model(model, X_fit, X_calib, X_test, y_test):
    model.fit(X_fit)

    calib_scores = model.decision_function(X_calib)

    threshold = float(np.quantile(calib_scores, TARGET_FPR))

    test_scores = model.decision_function(X_test)

    y_pred = (test_scores < threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test, y_pred
    ).ravel()

    actual_fpr = fp / (fp + tn)

    precision = precision_score(
        y_test, y_pred, zero_division=0
    )

    recall = recall_score(
        y_test, y_pred, zero_division=0
    )

    f1 = f1_score(
        y_test, y_pred, zero_division=0
    )

    attack_scores = -test_scores

    roc_auc = roc_auc_score(
        y_test, attack_scores
    )

    pr_auc = average_precision_score(
        y_test, attack_scores
    )

    return {
        "target_fpr": TARGET_FPR,
        "actual_test_fpr": float(actual_fpr),
        "threshold": threshold,
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

X_train = train_df.drop(columns=DROP_COLUMNS)
y_train = train_df["label"].astype(int)

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df["label"].astype(int)

# Train only on NORMAL traffic
X_normal = X_train[y_train == 0]

X_fit, X_calib = train_test_split(
    X_normal,
    test_size=0.25,
    random_state=RANDOM_STATE,
)

print(f"Normal training records: {len(X_normal):,}")
print(f"Model-fit records:       {len(X_fit):,}")
print(f"Calibration records:     {len(X_calib):,}")

# Fixed preprocessing: Log1p + StandardScaler
print("\nApplying Log1p transformation...")

X_fit = apply_log1p(X_fit)
X_calib = apply_log1p(X_calib)
X_test_processed = apply_log1p(X_test)

print("Fitting preprocessor...")

preprocessor = build_preprocessor(X_fit)

X_fit_processed = preprocessor.fit_transform(X_fit)
X_calib_processed = preprocessor.transform(X_calib)
X_test_processed = preprocessor.transform(X_test_processed)

print(
    f"Processed dimensions: {X_fit_processed.shape}"
)


# ---------------------------------------------------------
# HYPERPARAMETER EXPERIMENTS
# ---------------------------------------------------------

experiments = [
    {
        "name": "baseline_config",
        "params": {
            "n_estimators": 800,
            "max_samples": 1.0,
            "max_features": 0.7,
        },
    },
    {
        "name": "more_trees",
        "params": {
            "n_estimators": 1200,
            "max_samples": 1.0,
            "max_features": 0.7,
        },
    },
    {
        "name": "max_features_05",
        "params": {
            "n_estimators": 800,
            "max_samples": 1.0,
            "max_features": 0.5,
        },
    },
    {
        "name": "max_features_09",
        "params": {
            "n_estimators": 800,
            "max_samples": 1.0,
            "max_features": 0.9,
        },
    },
    {
        "name": "max_features_10",
        "params": {
            "n_estimators": 800,
            "max_samples": 1.0,
            "max_features": 1.0,
        },
    },
    {
        "name": "max_samples_07",
        "params": {
            "n_estimators": 800,
            "max_samples": 0.7,
            "max_features": 0.7,
        },
    },
    {
        "name": "max_samples_05",
        "params": {
            "n_estimators": 800,
            "max_samples": 0.5,
            "max_features": 0.7,
        },
    },
]


results = []

for experiment in experiments:

    name = experiment["name"]
    params = experiment["params"]

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)

    print(params)

    start = time.time()

    model = IsolationForest(
        n_estimators=params["n_estimators"],
        max_samples=params["max_samples"],
        max_features=params["max_features"],
        contamination=0.01,
        bootstrap=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    result = evaluate_model(
        model,
        X_fit_processed,
        X_calib_processed,
        X_test_processed,
        y_test,
    )

    result["name"] = name
    result["params"] = params
    result["time_seconds"] = round(
        time.time() - start, 1
    )

    results.append(result)

    print("\nRESULTS")
    print("-" * 50)
    print(
        f"Actual Test FPR: {result['actual_test_fpr']:.2%}"
    )
    print(
        f"Precision:       {result['precision']:.4f}"
    )
    print(
        f"Recall:          {result['recall']:.4f}"
    )
    print(
        f"F1:              {result['f1']:.4f}"
    )
    print(
        f"ROC-AUC:         {result['roc_auc']:.4f}"
    )
    print(
        f"PR-AUC:          {result['pr_auc']:.4f}"
    )


# Sort by F1 for easy inspection
results_sorted = sorted(
    results,
    key=lambda x: x["f1"],
    reverse=True,
)

report = {
    "dataset": "UNSW-NB15",
    "experiment": "Isolation Forest hyperparameter tuning",
    "preprocessing": "log1p + StandardScaler + OneHotEncoder",
    "target_fpr": TARGET_FPR,
    "random_state": RANDOM_STATE,
    "results": results_sorted,
}

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)


print("\n\n" + "=" * 70)
print("FINAL HYPERPARAMETER COMPARISON")
print("=" * 70)

print(
    f"{'Experiment':25}"
    f"{'FPR':>9}"
    f"{'Precision':>12}"
    f"{'Recall':>10}"
    f"{'F1':>10}"
    f"{'ROC-AUC':>11}"
    f"{'PR-AUC':>10}"
)

for r in results_sorted:
    print(
        f"{r['name']:25}"
        f"{r['actual_test_fpr']:>8.2%}"
        f"{r['precision']:>12.4f}"
        f"{r['recall']:>10.4f}"
        f"{r['f1']:>10.4f}"
        f"{r['roc_auc']:>11.4f}"
        f"{r['pr_auc']:>10.4f}"
    )

print("\nReport created:")
print(REPORT_PATH)