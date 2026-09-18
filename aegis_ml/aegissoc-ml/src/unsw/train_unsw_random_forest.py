from pathlib import Path
import joblib
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "UNSW_NB15_training-set.csv"
TEST_PATH = ROOT / "data" / "UNSW_NB15_testing-set.csv"

REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = REPORT_DIR / "unsw_nb15_random_forest.json"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TARGET_FPR = 0.05

# Skewed non-negative numerical features identified
# during our UNSW-NB15 feature audit.
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

DROP_COLUMNS = ["id", "attack_cat", "label"]

CATEGORICAL_FEATURES = [
    "proto",
    "service",
    "state",
]


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading UNSW-NB15...")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

print(f"Training shape: {train_df.shape}")
print(f"Testing shape:  {test_df.shape}")


# ============================================================
# PREPARE X AND y
# ============================================================

# IMPORTANT:
# label is removed from X so the model cannot directly see
# whether a row is an attack.
#
# attack_cat is also removed because it directly describes
# the attack category.
#
# id is only an identifier, not behavioral information.

X = train_df.drop(columns=DROP_COLUMNS)
y = train_df["label"].astype(int)

X_test = test_df.drop(columns=DROP_COLUMNS)
y_test = test_df["label"].astype(int)


print("\nClass distribution in training:")
print(y.value_counts().sort_index())

print("\nClass distribution in testing:")
print(y_test.value_counts().sort_index())


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

# The official testing set remains completely untouched.
#
# 80% of official training data -> model training
# 20% of official training data -> threshold validation

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

print("\nSplit:")
print(f"Model training: {X_train.shape[0]}")
print(f"Validation:     {X_val.shape[0]}")
print(f"Official test:  {X_test.shape[0]}")


# ============================================================
# PREPROCESSING
# ============================================================

numeric_features = [
    column
    for column in X.columns
    if column not in CATEGORICAL_FEATURES
]

# Apply log1p only to the selected skewed non-negative features.
#
# log1p(x) = log(1 + x)
#
# This reduces the influence of extremely large values.

def apply_log1p(df):
    df = df.copy()

    for column in LOG_FEATURES:
        if column in df.columns:
            df[column] = np.log1p(df[column].clip(lower=0))

    return df


X_train = apply_log1p(X_train)
X_val = apply_log1p(X_val)
X_test_processed = apply_log1p(X_test)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            numeric_features,
        ),
        (
            "categorical",
            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "onehot",
                        OneHotEncoder(
                            handle_unknown="ignore",
                            sparse_output=False,
                        ),
                    ),
                ]
            ),
            CATEGORICAL_FEATURES,
        ),
    ]
)


print("\nFitting preprocessing...")

X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test_processed)

print(f"Processed training shape: {X_train_processed.shape}")
print(f"Processed validation shape: {X_val_processed.shape}")
print(f"Processed test shape:       {X_test_processed.shape}")


# ============================================================
# RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(X_train_processed, y_train)

# Save trained model and preprocessing pipeline
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump(
    model,
    MODEL_DIR / "unsw_random_forest.joblib"
)

joblib.dump(
    preprocessor,
    MODEL_DIR / "unsw_random_forest_preprocessor.joblib"
)

print("\nSaved Random Forest model:")
print(MODEL_DIR / "unsw_random_forest.joblib")

print("Saved preprocessing pipeline:")
print(MODEL_DIR / "unsw_random_forest_preprocessor.joblib")
print("Random Forest training complete.")


# ============================================================
# VALIDATION THRESHOLD
# ============================================================

print("\nCalibrating threshold using validation normals...")

# Random Forest probability:
#
# probability close to 0 -> normal
# probability close to 1 -> attack
#
# We only use NORMAL validation samples to determine
# the threshold corresponding to our desired 5% FPR.

val_probabilities = model.predict_proba(X_val_processed)[:, 1]

normal_val_probabilities = val_probabilities[y_val == 0]

# For a 5% FPR:
#
# 95% of normal samples should be below the threshold.
#
# Therefore we use the 95th percentile.

threshold = float(
    np.quantile(
        normal_val_probabilities,
        1 - TARGET_FPR,
    )
)

print(f"Target FPR: {TARGET_FPR:.2%}")
print(f"Chosen threshold: {threshold:.6f}")


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\nEvaluating on official UNSW-NB15 testing set...")

test_probabilities = model.predict_proba(X_test_processed)[:, 1]

# probability >= threshold -> attack
y_pred = (test_probabilities >= threshold).astype(int)


# ============================================================
# METRICS
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1],
).ravel()

actual_fpr = fp / (fp + tn) if (fp + tn) else 0.0

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
    test_probabilities,
)

pr_auc = average_precision_score(
    y_test,
    test_probabilities,
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("UNSW-NB15 RANDOM FOREST RESULTS")
print("=" * 60)

print(f"Threshold:       {threshold:.6f}")
print(f"Actual FPR:      {actual_fpr:.4%}")
print(f"Precision:       {precision:.4f}")
print(f"Recall:          {recall:.4f}")
print(f"F1:              {f1:.4f}")
print(f"ROC-AUC:         {roc_auc:.4f}")
print(f"PR-AUC:          {pr_auc:.4f}")

print("\nConfusion Matrix:")
print(f"TN: {tn}")
print(f"FP: {fp}")
print(f"FN: {fn}")
print(f"TP: {tp}")


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

# This tells us which processed features Random Forest
# considered important.

feature_names = preprocessor.get_feature_names_out()

importances = model.feature_importances_

importance_df = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": importances,
    }
).sort_values(
    "importance",
    ascending=False,
)

top_features = importance_df.head(20)

print("\nTop 20 features:")
print(top_features.to_string(index=False))


# ============================================================
# SAVE REPORT
# ============================================================

report = {
    "model": "RandomForestClassifier",
    "dataset": "UNSW-NB15",
    "training_rows": int(len(train_df)),
    "testing_rows": int(len(test_df)),
    "raw_features": int(len(DROP_COLUMNS) + len(X.columns)),
    "model_features_before_encoding": int(X.shape[1]),
    "processed_features": int(X_train_processed.shape[1]),

    "preprocessing": {
        "log1p": True,
        "log_features": LOG_FEATURES,
        "standard_scaler": True,
        "one_hot_encoding": True,
        "categorical_features": CATEGORICAL_FEATURES,
    },

    "split": {
        "training_fraction": 0.80,
        "validation_fraction": 0.20,
        "random_state": RANDOM_STATE,
        "official_test_used_for_final_evaluation_only": True,
    },

    "random_forest": {
        "n_estimators": 300,
        "max_features": "sqrt",
        "class_weight": "balanced_subsample",
        "random_state": RANDOM_STATE,
    },

    "threshold": {
        "target_fpr": TARGET_FPR,
        "threshold": threshold,
        "calibration_method": "95th percentile of normal validation attack probabilities",
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

    "top_20_features": [
        {
            "feature": str(row["feature"]),
            "importance": float(row["importance"]),
        }
        for _, row in top_features.iterrows()
    ],
}


with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)


print(f"\nReport saved to:")
print(REPORT_PATH)

print("\nDone.")