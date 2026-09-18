import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
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

X_TRAIN_PATH = BASE_DIR / "models/CIC/X_train.npy"
X_CAL_PATH = BASE_DIR / "models/CIC/X_calibration.npy"
X_TEST_PATH = BASE_DIR / "models/CIC/X_test.npy"
Y_TEST_PATH = BASE_DIR / "models/CIC/y_test.npy"

REPORT_PATH = BASE_DIR / "reports/CIC/cic2017_if_hyperparameter_sweep.json"

TARGET_FPR = 0.05

N_ESTIMATORS = [400, 800, 1200]
MAX_SAMPLES = [0.5, 1.0]
MAX_FEATURES = [0.5, 0.7, 1.0]


def main():

    print("Loading CIC-IDS2017 preprocessed data...")

    X_train = np.load(X_TRAIN_PATH)
    X_cal = np.load(X_CAL_PATH)
    X_test = np.load(X_TEST_PATH)
    y_test = np.load(Y_TEST_PATH)

    print(f"Training rows:    {len(X_train):,}")
    print(f"Calibration rows: {len(X_cal):,}")
    print(f"Test rows:        {len(X_test):,}")
    print()

    results = []

    total = (
        len(N_ESTIMATORS)
        * len(MAX_SAMPLES)
        * len(MAX_FEATURES)
    )

    experiment_number = 0

    for n_estimators in N_ESTIMATORS:

        for max_samples in MAX_SAMPLES:

            for max_features in MAX_FEATURES:

                experiment_number += 1

                print("=" * 70)
                print(
                    f"Experiment {experiment_number}/{total}"
                )
                print(
                    f"n_estimators={n_estimators}, "
                    f"max_samples={max_samples}, "
                    f"max_features={max_features}"
                )

                model = IsolationForest(
                    n_estimators=n_estimators,
                    max_samples=max_samples,
                    max_features=max_features,
                    contamination=0.01,
                    bootstrap=False,
                    random_state=42,
                    n_jobs=-1,
                )

                print("Training...")

                model.fit(X_train)

                # Higher = more normal
                cal_scores = model.decision_function(X_cal)
                test_scores = model.decision_function(X_test)

                # Threshold calibrated ONLY on Tuesday normal traffic
                threshold = np.percentile(
                    cal_scores,
                    TARGET_FPR * 100
                )

                y_pred = (
                    test_scores < threshold
                ).astype(int)

                tn, fp, fn, tp = confusion_matrix(
                    y_test,
                    y_pred,
                    labels=[0, 1]
                ).ravel()

                actual_fpr = (
                    fp / (fp + tn)
                    if (fp + tn)
                    else 0.0
                )

                accuracy = accuracy_score(
                    y_test,
                    y_pred
                )

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
                    "n_estimators": n_estimators,
                    "max_samples": max_samples,
                    "max_features": max_features,

                    "target_fpr": TARGET_FPR,
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
                    f"Threshold:  {threshold:.6f}"
                )
                print(
                    f"Actual FPR: {actual_fpr * 100:.2f}%"
                )
                print(
                    f"Precision:  {precision:.4f}"
                )
                print(
                    f"Recall:     {recall:.4f}"
                )
                print(
                    f"F1:         {f1:.4f}"
                )
                print(
                    f"ROC-AUC:    {roc_auc:.4f}"
                )
                print(
                    f"PR-AUC:     {pr_auc:.4f}"
                )

    # Sort ONLY for reporting convenience.
    # We are not declaring a final model yet.
    results_sorted = sorted(
        results,
        key=lambda x: x["f1"],
        reverse=True
    )

    report = {
        "experiment": (
            "CIC-IDS2017 Isolation Forest "
            "hyperparameter sweep"
        ),

        "training": (
            "Monday BENIGN only"
        ),

        "calibration": (
            "Tuesday BENIGN only"
        ),

        "test": (
            "Wednesday + Thursday + Friday all traffic"
        ),

        "target_fpr": TARGET_FPR,

        "parameters_tested": {
            "n_estimators": N_ESTIMATORS,
            "max_samples": MAX_SAMPLES,
            "max_features": MAX_FEATURES,
        },

        "results": results,

        "results_sorted_by_f1": results_sorted,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("HYPERPARAMETER SWEEP COMPLETE")
    print("=" * 70)

    print()
    print("Top 5 configurations by F1:")

    for i, result in enumerate(
        results_sorted[:5],
        start=1
    ):

        print(
            f"{i}. "
            f"trees={result['n_estimators']}, "
            f"max_samples={result['max_samples']}, "
            f"max_features={result['max_features']} | "
            f"F1={result['f1']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"Precision={result['precision']:.4f} | "
            f"FPR={result['actual_test_fpr'] * 100:.2f}%"
        )

    print()
    print(f"Report saved to:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()