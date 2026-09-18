import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from explain import ReasonCodeExplainer
from preprocessing import build_preprocessor, get_feature_names, load_dataset


RANDOM_STATE = 42

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)

COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]


print("=" * 70)
print("SAVING TUNED ISOLATION FOREST")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD TRAINING DATA
# ------------------------------------------------------------

train, test = load_dataset(
    str(DATA_DIR / "KDDTrain+.txt"),
    str(DATA_DIR / "KDDTest+.txt"),
)

normal = train[train["is_anomaly"] == 0].reset_index(drop=True)
attack_train = train[train["is_anomaly"] == 1].reset_index(drop=True)

print(f"Total training rows: {len(train)}")
print(f"Normal training rows: {len(normal)}")
print(f"Attack training rows held out from fitting: {len(attack_train)}")


# ------------------------------------------------------------
# 2. SPLIT NORMAL DATA
# ------------------------------------------------------------

fit_normal, calibration_normal = train_test_split(
    normal,
    test_size=0.25,
    random_state=RANDOM_STATE
)

print(f"Model-fit normal rows: {len(fit_normal)}")
print(f"Calibration normal rows: {len(calibration_normal)}")


# ------------------------------------------------------------
# 3. PREPROCESS
# ------------------------------------------------------------

preprocessor = build_preprocessor()

X_fit = preprocessor.fit_transform(fit_normal)
X_calibration = preprocessor.transform(calibration_normal)

print(f"Feature space: {X_fit.shape[1]}")


# ------------------------------------------------------------
# 4. TUNED ISOLATION FOREST
# ------------------------------------------------------------

print("\nTraining tuned model...")

model = IsolationForest(
    n_estimators=800,
    max_samples=1.0,
    max_features=0.7,
    contamination=0.01,
    bootstrap=False,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_fit)

print("Training complete.")


# ------------------------------------------------------------
# 5. CALIBRATE 5% TARGET FPR
# ------------------------------------------------------------

calibration_scores = model.decision_function(
    X_calibration
)

threshold = float(
    np.quantile(
        calibration_scores,
        0.05
    )
)

print(f"\nOperating threshold: {threshold:.6f}")

# Use the same percentile bounds as train_model.py for the 0-100 risk scale.
calibration_scores_for_scaling = np.concatenate([
    calibration_scores,
    model.decision_function(
        preprocessor.transform(
            attack_train.sample(n=min(3000, len(attack_train)), random_state=RANDOM_STATE)
        )
    ),
])
score_low = float(np.percentile(calibration_scores_for_scaling, 1))
score_high = float(np.percentile(calibration_scores_for_scaling, 99))

threshold_table = {
    f"fpr_{int(fpr * 100)}pct": float(np.quantile(calibration_scores, fpr))
    for fpr in (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10)
}

print(f"Risk-score calibration bounds: low={score_low:.6f}, high={score_high:.6f}")

# Evaluate only after fitting and calibration are complete. KDDTest+ is never
# used by the preprocessor, model, threshold, or hyperparameter selection.
X_test = preprocessor.transform(test)
test_scores = model.decision_function(X_test)
y_true = test["is_anomaly"].values
y_pred = (test_scores < threshold).astype(int)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="binary", zero_division=0
)
accuracy = float((y_pred == y_true).mean())
anomaly_likelihood = -test_scores
roc_auc = roc_auc_score(y_true, anomaly_likelihood)
pr_auc = average_precision_score(y_true, anomaly_likelihood)
cm = confusion_matrix(y_true, y_pred)

sweep_rows = []
for name, threshold_value in threshold_table.items():
    sweep_prediction = (test_scores < threshold_value).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, sweep_prediction).ravel()
    sweep_precision, sweep_recall, sweep_f1, _ = precision_recall_fscore_support(
        y_true, sweep_prediction, average="binary", zero_division=0
    )
    sweep_rows.append({
        "operating_point": name,
        "threshold": threshold_value,
        "precision": sweep_precision,
        "recall": sweep_recall,
        "f1": sweep_f1,
        "flag_rate": float(sweep_prediction.mean()),
        "fpr_actual": float(fp / (fp + tn)),
    })

recall_by_category = {}
for category in ["dos", "probe", "r2l", "u2r"]:
    mask = test["attack_category"] == category
    if mask.sum() > 0:
        recall_by_category[category] = float(y_pred[mask.values].mean())

train_labels = set(train["label"].unique())
novel_mask = (
    (test["is_anomaly"] == 1)
    & (~test["label"].isin(train_labels))
).values
novel_labels = sorted(test.loc[novel_mask, "label"].unique().tolist())
recall_novel = float(y_pred[novel_mask].mean()) if novel_mask.sum() > 0 else None

test_metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "confusion_matrix": cm.tolist(),
    "recall_by_attack_family": recall_by_category,
    "recall_on_novel_unseen_attack_types": recall_novel,
    "novel_attack_types_in_test_set": novel_labels,
}

print("\n--- Untouched KDDTest+ evaluation ---")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")


# ------------------------------------------------------------
# 6. GET FEATURE NAMES
# ------------------------------------------------------------

feature_names = get_feature_names(preprocessor)


# ------------------------------------------------------------
# 7. SAVE MODEL + PREPROCESSOR
# ------------------------------------------------------------

model_path = MODEL_DIR / "isolation_forest.joblib"
preprocessor_path = MODEL_DIR / "preprocessor.joblib"

joblib.dump(
    model,
    model_path
)

joblib.dump(
    preprocessor,
    preprocessor_path
)

reason_explainer = ReasonCodeExplainer().fit(fit_normal)
joblib.dump(reason_explainer, MODEL_DIR / "reason_explainer.joblib")

print("\nSaved:")
print(model_path)
print(preprocessor_path)


# ------------------------------------------------------------
# 8. CREATE CORRECT METADATA
# ------------------------------------------------------------

metadata = {
    "model_type": "IsolationForest",

    "sklearn_training":
        "semi-supervised (trained on NORMAL traffic only)",

    "dataset":
        "NSL-KDD (Canadian Institute for Cybersecurity)",

    "n_estimators": 800,

    "max_samples": 1.0,

    "max_features": 0.7,

    "contamination": 0.01,

    "bootstrap": False,

    "random_state": RANDOM_STATE,

    "feature_count": int(X_fit.shape[1]),

    "feature_names": feature_names,

    "operating_threshold": threshold,

    "target_fpr": 0.05,

    "threshold_table": threshold_table,

    "risk_score_calibration": {
        "score_low": score_low,
        "score_high": score_high,
    },

    "test_metrics": test_metrics,

    "threshold_sweep": sweep_rows,

    "training_rows_used": int(len(fit_normal)),

    "calibration_rows_used": int(len(calibration_normal)),

    "notes": [
        "Tuned Isolation Forest selected through hyperparameter search.",
        "Model trained only on normal traffic.",
        "Threshold calibrated on held-out normal calibration data.",
        "KDDTest+ was used only for final evaluation after fitting and calibration."
    ]
}


metadata_path = MODEL_DIR / "metadata.json"

with open(
    metadata_path,
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        metadata,
        f,
        indent=2
    )

print(metadata_path)


# ------------------------------------------------------------
# 9. VERIFY IMMEDIATELY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VERIFYING SAVED ARTIFACTS")
print("=" * 70)

saved_model = joblib.load(
    model_path
)

saved_preprocessor = joblib.load(
    preprocessor_path
)

with open(
    metadata_path,
    "r",
    encoding="utf-8"
) as f:
    saved_metadata = json.load(f)


print(
    f"trees:          {saved_model.n_estimators}"
)

print(
    f"max_samples:    {saved_model.max_samples}"
)

print(
    f"max_features:   {saved_model.max_features}"
)

print(
    f"contamination:   {saved_model.contamination}"
)

print(
    f"bootstrap:       {saved_model.bootstrap}"
)

print(
    f"feature_count:   {saved_metadata['feature_count']}"
)

print(
    f"threshold:       {saved_metadata['operating_threshold']:.6f}"
)

print(
    f"risk calibration: [{saved_metadata['risk_score_calibration']['score_low']:.6f}, "
    f"{saved_metadata['risk_score_calibration']['score_high']:.6f}]"
)

print(
    f"test metrics:    {len(saved_metadata['test_metrics'])} fields"
)

print(
    f"reason explainer: {(MODEL_DIR / 'reason_explainer.joblib').exists()}"
)

print("\nDONE.")