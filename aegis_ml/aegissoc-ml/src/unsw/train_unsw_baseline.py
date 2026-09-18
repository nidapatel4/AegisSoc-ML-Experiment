from pathlib import Path
import json
import time

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "data" / "UNSW_NB15_training-set.csv"
TEST_PATH = BASE_DIR / "data" / "UNSW_NB15_testing-set.csv"

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = REPORTS_DIR / "unsw_nb15_baseline_comparison.json"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TARGET_FPR = 0.05

N_ESTIMATORS = 800
MAX_SAMPLES = 1.0
MAX_FEATURES = 0.7
CONTAMINATION = 0.01


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
# LOAD DATA
# ============================================================

print("\nLoading UNSW-NB15...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Training rows: {len(train_df):,}")
print(f"Testing rows:  {len(test_df):,}")


# ============================================================
# PREPARE X / y
# ============================================================

X_train_full = train_df.drop(columns=DROP_COLUMNS)
y_train_full = train_df["label"]

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df["label"]


# ============================================================
# NORMAL-ONLY TRAINING
# ============================================================

normal_mask = y_train_full == 0

X_normal = X_train_full.loc[normal_mask].copy()

print(f"\nNormal training records: {len(X_normal):,}")

X_fit, X_calib = train_test_split(
    X_normal,
    test_size=0.25,
    random_state=RANDOM_STATE,
)

print(f"Model-fit normal records: {len(X_fit):,}")
print(f"Calibration normal records: {len(X_calib):,}")


# ============================================================
# NUMERICAL FEATURES
# ============================================================

FEATURE_COLUMNS = list(X_train_full.columns)

NUMERICAL_COLUMNS = [
    column
    for column in FEATURE_COLUMNS
    if column not in CATEGORICAL_COLUMNS
]


# ============================================================
# PREPROCESSOR
# ============================================================

def create_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERICAL_COLUMNS,
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


# ============================================================
# LOG TRANSFORMATION
# ============================================================

def apply_log_transform(df):
    df = df.copy()

    for column in SKEWED_COLUMNS:
        df[column] = np.log1p(df[column])

    return df


# ============================================================
# TRAIN + EVALUATE
# ============================================================

def run_experiment(name, use_log_transform):

    print("\n" + "=" * 70)
    print(f"EXPERIMENT: {name}")
    print("=" * 70)

    start_time = time.time()

    # --------------------------------------------------------
    # Copy datasets
    # --------------------------------------------------------

    fit_data = X_fit.copy()
    calib_data = X_calib.copy()
    test_data = X_test.copy()

    # --------------------------------------------------------
    # Optional log transformation
    # --------------------------------------------------------

    if use_log_transform:

        print("\nApplying log1p transformation...")

        fit_data = apply_log_transform(fit_data)
        calib_data = apply_log_transform(calib_data)
        test_data = apply_log_transform(test_data)

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    print("Fitting preprocessor...")

    preprocessor = create_preprocessor()

    X_fit_processed = preprocessor.fit_transform(fit_data)

    X_calib_processed = preprocessor.transform(calib_data)

    X_test_processed = preprocessor.transform(test_data)

    print(
        f"Processed dimensions: "
        f"{X_fit_processed.shape}"
    )

    # --------------------------------------------------------
    # Isolation Forest
    # --------------------------------------------------------

    print("\nTraining Isolation Forest...")

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        max_samples=MAX_SAMPLES,
        max_features=MAX_FEATURES,
        contamination=CONTAMINATION,
        bootstrap=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_fit_processed)

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    print("Calibrating threshold...")

    calib_scores = model.decision_function(
        X_calib_processed
    )

    threshold = float(
        np.quantile(
            calib_scores,
            TARGET_FPR,
        )
    )

    print(
        f"Target FPR: {TARGET_FPR * 100:.1f}%"
    )

    print(
        f"Threshold: {threshold:.6f}"
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    print("Evaluating test set...")

    test_scores = model.decision_function(
        X_test_processed
    )

    y_pred = (
        test_scores < threshold
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
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

    # Isolation Forest gives anomaly scores where
    # lower = more anomalous.
    # Negating makes higher = more attack-like,
    # which is suitable for ROC-AUC / PR-AUC.
    attack_scores = -test_scores

    roc_auc = roc_auc_score(
        y_test,
        attack_scores,
    )

    pr_auc = average_precision_score(
        y_test,
        attack_scores,
    )

    elapsed = time.time() - start_time

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\nRESULTS")
    print("-" * 50)

    print(f"Target FPR:       {TARGET_FPR * 100:.2f}%")
    print(f"Actual Test FPR:  {actual_fpr * 100:.2f}%")
    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")
    print(f"F1:               {f1:.4f}")
    print(f"ROC-AUC:          {roc_auc:.4f}")
    print(f"PR-AUC:           {pr_auc:.4f}")

    print("\nConfusion Matrix")
    print(
        f"TN={tn}, FP={fp}, "
        f"FN={fn}, TP={tp}"
    )

    print(f"\nTime: {elapsed:.1f} seconds")

    return {
        "preprocessing": name,

        "log_transform": use_log_transform,

        "input_features": len(FEATURE_COLUMNS),

        "numerical_features": len(
            NUMERICAL_COLUMNS
        ),

        "categorical_features": len(
            CATEGORICAL_COLUMNS
        ),

        "output_dimensions": int(
            X_fit_processed.shape[1]
        ),

        "model": {
            "algorithm": "IsolationForest",
            "n_estimators": N_ESTIMATORS,
            "max_samples": MAX_SAMPLES,
            "max_features": MAX_FEATURES,
            "contamination": CONTAMINATION,
            "random_state": RANDOM_STATE,
        },

        "calibration": {
            "normal_fit_rows": len(X_fit),
            "normal_calibration_rows": len(X_calib),
            "target_fpr": TARGET_FPR,
            "threshold": threshold,
        },

        "test": {
            "rows": len(X_test),
            "normal": int((y_test == 0).sum()),
            "attack": int((y_test == 1).sum()),
        },

        "metrics": {
            "actual_fpr": actual_fpr,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        },

        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },

        "training_time_seconds": elapsed,
    }


# ============================================================
# RUN BOTH EXPERIMENTS
# ============================================================

results = []

results.append(
    run_experiment(
        "standard_scaler",
        use_log_transform=False,
    )
)

results.append(
    run_experiment(
        "log1p_plus_standard_scaler",
        use_log_transform=True,
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

report = {
    "dataset": "UNSW-NB15",

    "training_file": TRAIN_PATH.name,

    "testing_file": TEST_PATH.name,

    "methodology": {
        "normal_only_training": True,
        "normal_split": "75% fit / 25% calibration",
        "random_state": RANDOM_STATE,
        "target_calibration_fpr": TARGET_FPR,
    },

    "experiments": results,
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
# FINAL COMPARISON
# ============================================================

print("\n\n" + "=" * 70)
print("FINAL BASELINE COMPARISON")
print("=" * 70)

print(
    f"{'Preprocessing':<30}"
    f"{'FPR':>10}"
    f"{'Precision':>12}"
    f"{'Recall':>10}"
    f"{'F1':>10}"
    f"{'ROC-AUC':>12}"
    f"{'PR-AUC':>10}"
)

for result in results:

    metrics = result["metrics"]

    print(
        f"{result['preprocessing']:<30}"
        f"{metrics['actual_fpr'] * 100:>9.2f}%"
        f"{metrics['precision']:>12.4f}"
        f"{metrics['recall']:>10.4f}"
        f"{metrics['f1']:>10.4f}"
        f"{metrics['roc_auc']:>12.4f}"
        f"{metrics['pr_auc']:>10.4f}"
    )

print("\nReport created:")
print(OUTPUT_PATH)