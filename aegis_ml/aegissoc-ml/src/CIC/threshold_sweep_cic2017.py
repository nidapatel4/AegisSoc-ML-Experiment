import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models/CIC/cic2017_isolation_forest_baseline.joblib"
X_CAL_PATH = BASE_DIR / "models/CIC/X_calibration.npy"
X_TEST_PATH = BASE_DIR / "models/CIC/X_test.npy"
Y_TEST_PATH = BASE_DIR / "models/CIC/y_test.npy"

REPORT_PATH = BASE_DIR / "reports/CIC/cic2017_threshold_sweep.json"


# Target operating points
TARGET_FPRS = np.arange(0.01, 0.11, 0.01)


def main():
    print("Loading model and data...")

    model = joblib.load(MODEL_PATH)
    X_cal = np.load(X_CAL_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    print(f"Calibration rows: {len(X_cal):,}")
    print(f"Test rows:        {len(X_test):,}")
    print()

    # Isolation Forest:
    # higher decision_function = more normal
    # lower decision_function = more anomalous
    cal_scores = model.decision_function(X_cal)
    test_scores = model.decision_function(X_test)

    results = []

    for target_fpr in TARGET_FPRS:

        # Threshold chosen ONLY from calibration normals.
        # Example: 5% FPR means 5% of calibration normals
        # should be classified as anomalies.
        threshold = np.percentile(
            cal_scores,
            target_fpr * 100
        )

        y_pred = (test_scores < threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred,
            labels=[0, 1]
        ).ravel()

        actual_fpr = fp / (fp + tn) if (fp + tn) else 0.0

        accuracy = accuracy_score(y_test, y_pred)
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

        roc_auc = roc_auc_score(
            y_test,
            -test_scores
        )

        pr_auc = average_precision_score(
            y_test,
            -test_scores
        )

        result = {
            "target_fpr": round(float(target_fpr), 4),
            "threshold": float(threshold),

            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),

            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),

            "actual_test_fpr": float(actual_fpr),

            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

        results.append(result)

        print(
            f"Target FPR: {target_fpr * 100:>2.0f}% | "
            f"Threshold: {threshold:.6f} | "
            f"Actual FPR: {actual_fpr * 100:>6.2f}% | "
            f"Precision: {precision:.4f} | "
            f"Recall: {recall:.4f} | "
            f"F1: {f1:.4f}"
        )

    # Find highest F1 purely for reporting.
    # This does NOT mean we automatically select it as the final operating point.
    best_f1 = max(results, key=lambda x: x["f1"])

    report = {
        "experiment": "CIC-IDS2017 Isolation Forest threshold sweep",

        "model": str(MODEL_PATH),

        "calibration": {
            "dataset": "Tuesday BENIGN only",
            "rows": int(len(X_cal)),
            "used_for": "threshold calibration only",
        },

        "test": {
            "dataset": "Wednesday + Thursday + Friday all traffic",
            "rows": int(len(X_test)),
            "used_for": "final evaluation only",
        },

        "important_rule": (
            "Thresholds were calibrated using Tuesday normal traffic "
            "only. Untouched test data was not used for threshold selection."
        ),

        "results": results,

        "highest_f1_operating_point": best_f1,
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print()
    print("=" * 70)
    print("Threshold sweep complete.")
    print(f"Report saved to: {REPORT_PATH}")
    print("=" * 70)

    print()
    print("Highest-F1 operating point:")
    print(
        f"Target FPR:    {best_f1['target_fpr'] * 100:.0f}%"
    )
    print(
        f"Threshold:     {best_f1['threshold']:.6f}"
    )
    print(
        f"Actual FPR:    {best_f1['actual_test_fpr'] * 100:.2f}%"
    )
    print(
        f"Precision:     {best_f1['precision']:.4f}"
    )
    print(
        f"Recall:        {best_f1['recall']:.4f}"
    )
    print(
        f"F1:            {best_f1['f1']:.4f}"
    )


if __name__ == "__main__":
    main()