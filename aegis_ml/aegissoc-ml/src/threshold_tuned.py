import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))

from preprocessing import build_preprocessor


RANDOM_STATE = 42

DATA_DIR = Path("data")
TRAIN_PATH = DATA_DIR / "KDDTrain+.txt"
TEST_PATH = DATA_DIR / "KDDTest+.txt"

TARGET_FPRS = [
    0.01, 0.02, 0.03, 0.04, 0.05,
    0.06, 0.07, 0.08, 0.09, 0.10
]


# ---------------------------------------------------------
# NSL-KDD columns
# ---------------------------------------------------------
COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]


# ---------------------------------------------------------
# Load NSL-KDD
# ---------------------------------------------------------
train_df = pd.read_csv(
    TRAIN_PATH,
    header=None,
    names=COLUMNS
)

test_df = pd.read_csv(
    TEST_PATH,
    header=None,
    names=COLUMNS
)

print(f"Train: {train_df.shape}")
print(f"Test : {test_df.shape}")


# ---------------------------------------------------------
# Normal vs attack
# ---------------------------------------------------------
normal_train_df = train_df[
    train_df["label"] == "normal"
].copy()

attack_train_df = train_df[
    train_df["label"] != "normal"
].copy()

print(f"Normal train: {len(normal_train_df)}")
print(f"Attack train: {len(attack_train_df)}")


# ---------------------------------------------------------
# Split normal data:
# 75% model fitting
# 25% threshold calibration
# ---------------------------------------------------------
fit_normal_df, calib_normal_df = train_test_split(
    normal_train_df,
    test_size=0.25,
    random_state=RANDOM_STATE
)


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
preprocessor = build_preprocessor()

X_fit = preprocessor.fit_transform(fit_normal_df)
X_calib = preprocessor.transform(calib_normal_df)
X_test = preprocessor.transform(test_df)

print(f"Feature space: {X_fit.shape[1]}")


# ---------------------------------------------------------
# Tuned Isolation Forest
# ---------------------------------------------------------
model = IsolationForest(
    n_estimators=800,
    max_samples=1.0,
    max_features=0.7,
    contamination=0.01,
    bootstrap=False,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

print("\nTraining tuned Isolation Forest...")
model.fit(X_fit)

print("Training complete.")


# ---------------------------------------------------------
# Scores
# ---------------------------------------------------------
calib_scores = model.decision_function(X_calib)
test_scores = model.decision_function(X_test)

y_test = (
    test_df["label"] != "normal"
).astype(int).to_numpy()


# ---------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------
print("\n" + "=" * 100)
print("TUNED ISOLATION FOREST — THRESHOLD ANALYSIS")
print("=" * 100)

print(
    f"{'Target FPR':>12} "
    f"{'Threshold':>12} "
    f"{'Actual FPR':>12} "
    f"{'Precision':>12} "
    f"{'Recall':>10} "
    f"{'F1':>10}"
)

print("-" * 100)

results = []

for target_fpr in TARGET_FPRS:

    threshold = np.quantile(
        calib_scores,
        target_fpr
    )

    y_pred = (
        test_scores < threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1]
    ).ravel()

    actual_fpr = fp / (fp + tn)

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

    print(
        f"{target_fpr * 100:11.0f}% "
        f"{threshold:12.6f} "
        f"{actual_fpr * 100:11.2f}% "
        f"{precision:12.4f} "
        f"{recall:10.4f} "
        f"{f1:10.4f}"
    )

    results.append({
        "target_fpr": target_fpr,
        "threshold": threshold,
        "actual_fpr": actual_fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    })


# ---------------------------------------------------------
# AUC
# ---------------------------------------------------------
roc_auc = roc_auc_score(
    y_test,
    -test_scores
)

pr_auc = average_precision_score(
    y_test,
    -test_scores
)

print("\n" + "=" * 100)
print("AUC")
print("=" * 100)

print(f"ROC-AUC : {roc_auc:.4f}")
print(f"PR-AUC  : {pr_auc:.4f}")


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
reports_dir = Path("reports")
reports_dir.mkdir(exist_ok=True)

output_path = reports_dir / "tuned_threshold_analysis.json"

pd.DataFrame(results).to_json(
    output_path,
    orient="records",
    indent=2
)

print("\nSaved:")
print(output_path)
