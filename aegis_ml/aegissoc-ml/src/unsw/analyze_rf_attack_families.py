from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "UNSW_NB15_training-set.csv"
TEST_PATH = ROOT / "data" / "UNSW_NB15_testing-set.csv"

MODEL_PATH = ROOT / "models" / "unsw_random_forest.joblib"
PREPROCESSOR_PATH = ROOT / "models" / "unsw_random_forest_preprocessor.joblib"

REPORT_PATH = ROOT / "reports" / "unsw_nb15_random_forest_attack_family.json"

SKEWED_COLUMNS = [
    "spkts", "dpkts", "sbytes", "dbytes", "sloss", "dloss",
    "dinpkt", "sjit", "djit", "trans_depth",
    "response_body_len", "ct_flw_http_mthd"
]

THRESHOLD = 0.65

print("Loading saved Random Forest...")
model = joblib.load(MODEL_PATH)

print("Loading saved preprocessor...")
preprocessor = joblib.load(PREPROCESSOR_PATH)

print("\nLoading UNSW-NB15 datasets...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Recreate exact training split used during RF training
from sklearn.model_selection import train_test_split

X_train_full = train.drop(columns=["id", "attack_cat", "label"])
y_train_full = train["label"].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.20,
    stratify=y_train_full,
    random_state=42,
)

# Prepare validation data
for col in SKEWED_COLUMNS:
    X_val[col] = np.log1p(X_val[col].clip(lower=0))

# Prepare test data
X_test = test.drop(columns=["id", "attack_cat", "label"])

for col in SKEWED_COLUMNS:
    X_test[col] = np.log1p(X_test[col].clip(lower=0))

print("\nTransforming data...")

X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print(f"Validation shape: {X_val_processed.shape}")
print(f"Test shape:       {X_test_processed.shape}")

print("\nGenerating RF probabilities...")

val_prob = model.predict_proba(X_val_processed)[:, 1]
test_prob = model.predict_proba(X_test_processed)[:, 1]

# ---------------------------------------------------------
# Overall test metrics at fixed threshold
# ---------------------------------------------------------

test_predictions = (test_prob >= THRESHOLD).astype(int)

from sklearn.metrics import confusion_matrix

tn, fp, fn, tp = confusion_matrix(
    test["label"],
    test_predictions,
    labels=[0, 1]
).ravel()

overall_fpr = fp / (fp + tn)

print("\n" + "=" * 70)
print(f"OVERALL TEST RESULT — THRESHOLD {THRESHOLD}")
print("=" * 70)

print(f"FPR: {overall_fpr * 100:.2f}%")
print(f"TN:  {tn}")
print(f"FP:  {fp}")
print(f"FN:  {fn}")
print(f"TP:  {tp}")


# ---------------------------------------------------------
# Attack-family recall
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("ATTACK FAMILY RECALL")
print("=" * 70)

attack_families = []

attack_rows = test[test["label"] == 1]

for family in sorted(attack_rows["attack_cat"].dropna().unique()):

    family_mask = (
        (test["label"] == 1) &
        (test["attack_cat"] == family)
    )

    total = int(family_mask.sum())

    detected = int(
        ((test_predictions == 1) & family_mask).sum()
    )

    missed = total - detected

    recall = detected / total if total else 0

    result = {
        "attack_family": family,
        "total": total,
        "detected": detected,
        "missed": missed,
        "recall": recall,
    }

    attack_families.append(result)

    print(
        f"{family:20s} | "
        f"{detected:5d}/{total:<5d} | "
        f"Recall {recall * 100:6.2f}%"
    )


# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

report = {
    "dataset": "UNSW-NB15",
    "model": "Random Forest",
    "threshold": THRESHOLD,
    "overall": {
        "fpr": overall_fpr,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    },
    "attack_family_recall": attack_families,
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 70)
print("ATTACK FAMILY ANALYSIS COMPLETE")
print("=" * 70)

print("\nReport saved to:")
print(REPORT_PATH)

print("\nDone.")
