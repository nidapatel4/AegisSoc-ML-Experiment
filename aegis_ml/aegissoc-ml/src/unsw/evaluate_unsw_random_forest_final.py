from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]

TEST_PATH = ROOT / "data" / "UNSW_NB15_testing-set.csv"
MODEL_PATH = ROOT / "models" / "unsw_random_forest.joblib"
PREPROCESSOR_PATH = ROOT / "models" / "unsw_random_forest_preprocessor.joblib"
REPORT_PATH = ROOT / "reports" / "unsw_nb15_random_forest_final.json"

SKEWED_COLUMNS = [
    "spkts", "dpkts", "sbytes", "dbytes", "sloss", "dloss",
    "dinpkt", "sjit", "djit", "trans_depth",
    "response_body_len", "ct_flw_http_mthd"
]

print("Loading saved Random Forest...")
model = joblib.load(MODEL_PATH)

print("Loading saved preprocessor...")
preprocessor = joblib.load(PREPROCESSOR_PATH)

print("\nLoading official UNSW-NB15 testing set...")
df = pd.read_csv(TEST_PATH)

y_test = df["label"].astype(int)

X_test = df.drop(columns=["id", "attack_cat", "label"])

print("\nApplying Log1p transformation...")

for col in SKEWED_COLUMNS:
    X_test[col] = np.log1p(X_test[col].clip(lower=0))

print("\nTransforming test data...")
X_test_processed = preprocessor.transform(X_test)

print(f"Processed test shape: {X_test_processed.shape}")

print("\nGenerating attack probabilities...")
probabilities = model.predict_proba(X_test_processed)[:, 1]

thresholds = [0.50, 0.65]

results = []

print("\n" + "=" * 70)
print("FINAL RANDOM FOREST TEST EVALUATION")
print("=" * 70)

for threshold in thresholds:

    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    ).ravel()

    fpr = fp / (fp + tn)

    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    roc_auc = roc_auc_score(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)

    result = {
        "threshold": threshold,
        "fpr": fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    results.append(result)

    print(f"\nThreshold: {threshold:.2f}")
    print(f"FPR:       {fpr * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"TN: {tn} | FP: {fp} | FN: {fn} | TP: {tp}")


report = {
    "dataset": "UNSW-NB15",
    "experiment": "Final Random Forest evaluation",
    "note": "Official test set evaluated after threshold selection on validation data.",
    "thresholds_evaluated": results
}

REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

with open(REPORT_PATH, "w") as f:
    json.dump(report, f, indent=2)

print("\n" + "=" * 70)
print("FINAL EVALUATION COMPLETE")
print("=" * 70)

print("\nReport saved to:")
print(REPORT_PATH)

print("\nDone.")