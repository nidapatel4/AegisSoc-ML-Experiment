from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score


ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "UNSW_NB15_training-set.csv"

MODEL_PATH = ROOT / "models" / "unsw_random_forest.joblib"
PREPROCESSOR_PATH = ROOT / "models" / "unsw_random_forest_preprocessor.joblib"

REPORT_PATH = ROOT / "reports" / "unsw_nb15_random_forest_validation_threshold.json"


# Same columns used during RF training
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


print("Loading saved Random Forest...")
model = joblib.load(MODEL_PATH)
print("Model loaded.")

print("Loading saved preprocessor...")
preprocessor = joblib.load(PREPROCESSOR_PATH)
print("Preprocessor loaded.")


print("\nLoading UNSW-NB15 training set...")
df = pd.read_csv(TRAIN_PATH)

# Separate labels before preprocessing
y = df["label"].astype(int)

X = df.drop(columns=["id", "attack_cat", "label"])


# ---------------------------------------------------------
# Recreate the exact Log1p transformation used during
# Random Forest training
# ---------------------------------------------------------

print("\nApplying Log1p transformation...")

for col in SKEWED_COLUMNS:
    X[col] = np.log1p(X[col].clip(lower=0))


# ---------------------------------------------------------
# Recreate the SAME 80/20 stratified split used during
# Random Forest training
# ---------------------------------------------------------

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

print(f"Training rows:   {len(X_train)}")
print(f"Validation rows: {len(X_val)}")


# ---------------------------------------------------------
# IMPORTANT:
# The saved preprocessor was fitted on X_train.
# We therefore ONLY transform validation data here.
# ---------------------------------------------------------

print("\nTransforming validation data...")

X_val_processed = preprocessor.transform(X_val)

print(f"Processed validation shape: {X_val_processed.shape}")


# ---------------------------------------------------------
# Generate probabilities ONCE
# ---------------------------------------------------------

print("\nGenerating validation attack probabilities...")

val_probabilities = model.predict_proba(X_val_processed)[:, 1]

print("Probabilities generated.")


# ---------------------------------------------------------
# Use ONLY NORMAL validation samples for threshold
# calibration.
#
# Target:
# 5% validation false-positive rate.
# ---------------------------------------------------------

normal_scores = val_probabilities[y_val == 0]

target_fpr = 0.05

calibrated_threshold = np.quantile(
    normal_scores,
    1 - target_fpr
)

print("\n" + "=" * 70)
print("VALIDATION THRESHOLD CALIBRATION")
print("=" * 70)

print(f"Target FPR:          {target_fpr * 100:.2f}%")
print(f"Calibrated threshold: {calibrated_threshold:.6f}")


# ---------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------

thresholds = [
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
]

results = []

print("\nRunning validation threshold sweep...\n")

for threshold in thresholds:

    predictions = (
        val_probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1],
    ).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0
    precision = precision_score(
        y_val,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        y_val,
        predictions,
        zero_division=0,
    )
    f1 = f1_score(
        y_val,
        predictions,
        zero_division=0,
    )

    results.append({
        "threshold": float(threshold),
        "fpr": float(fpr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    })

    print(
        f"Threshold {threshold:.2f} | "
        f"FPR {fpr * 100:.2f}% | "
        f"Precision {precision:.4f} | "
        f"Recall {recall:.4f} | "
        f"F1 {f1:.4f}"
    )


# ---------------------------------------------------------
# Find best F1 on VALIDATION ONLY
# ---------------------------------------------------------

best_f1 = max(results, key=lambda x: x["f1"])

closest_to_5 = min(
    results,
    key=lambda x: abs(x["fpr"] - target_fpr)
)


print("\n" + "=" * 70)
print("VALIDATION THRESHOLD SWEEP COMPLETE")
print("=" * 70)

print("\nBest validation F1:")
print(f"Threshold: {best_f1['threshold']:.2f}")
print(f"FPR:       {best_f1['fpr'] * 100:.2f}%")
print(f"Precision: {best_f1['precision']:.4f}")
print(f"Recall:    {best_f1['recall']:.4f}")
print(f"F1:        {best_f1['f1']:.4f}")

print("\nClosest to 5% validation FPR:")
print(f"Threshold: {closest_to_5['threshold']:.2f}")
print(f"FPR:       {closest_to_5['fpr'] * 100:.2f}%")
print(f"Precision: {closest_to_5['precision']:.4f}")
print(f"Recall:    {closest_to_5['recall']:.4f}")
print(f"F1:        {closest_to_5['f1']:.4f}")


# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

report = {
    "dataset": "UNSW-NB15",
    "experiment": "Random Forest validation threshold calibration",
    "split": {
        "method": "80/20 stratified",
        "random_state": 42,
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
    },
    "preprocessing": {
        "log1p": True,
        "standard_scaler": True,
        "one_hot_encoder": True,
    },
    "target_fpr": target_fpr,
    "calibrated_threshold": float(calibrated_threshold),
    "best_validation_f1": best_f1,
    "closest_to_target_fpr": closest_to_5,
    "threshold_sweep": results,
}

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)


print("\nReport saved to:")
print(REPORT_PATH)

print("\nDone.")