# import json
# from pathlib import Path

# import numpy as np
# from sklearn.compose import ColumnTransformer
# from sklearn.ensemble import IsolationForest
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.svm import OneClassSVM
# from sklearn.metrics import (
#     precision_score,
#     recall_score,
#     f1_score,
#     confusion_matrix,
#     average_precision_score,
#     roc_auc_score,
# )

# from preprocessing import (
#     load_dataset,
#     CATEGORICAL_FEATURES,
#     NUMERIC_FEATURES,
# )


# DATA_TRAIN = "data/KDDTrain+.txt"
# DATA_TEST = "data/KDDTest+.txt"

# TARGET_FPRS = [
#     0.01,
#     0.02,
#     0.03,
#     0.04,
#     0.05,
#     0.06,
#     0.07,
#     0.08,
#     0.09,
#     0.10,
# ]


# def build_preprocessor():
#     return ColumnTransformer(
#         transformers=[
#             (
#                 "cat",
#                 OneHotEncoder(handle_unknown="ignore"),
#                 CATEGORICAL_FEATURES,
#             ),
#             (
#                 "num",
#                 StandardScaler(),
#                 NUMERIC_FEATURES,
#             ),
#         ]
#     )


# def evaluate_thresholds(
#     model_name,
#     model,
#     X_fit,
#     X_calibration,
#     X_test,
#     y_test,
# ):
#     model.fit(X_fit)

#     calibration_scores = model.decision_function(
#         X_calibration
#     )
#     test_scores = model.decision_function(X_test)

#     # Isolation Forest:
#     # lower score = more anomalous
#     #
#     # One-Class SVM:
#     # lower score = more anomalous
#     #
#     # Therefore both use the same threshold direction.

#     results = []

#     for target_fpr in TARGET_FPRS:
#         threshold = float(
#             np.quantile(
#                 calibration_scores,
#                 target_fpr,
#             )
#         )

#         predictions = (
#             test_scores < threshold
#         ).astype(int)

#         precision = precision_score(
#             y_test,
#             predictions,
#             zero_division=0,
#         )

#         recall = recall_score(
#             y_test,
#             predictions,
#             zero_division=0,
#         )

#         f1 = f1_score(
#             y_test,
#             predictions,
#             zero_division=0,
#         )

#         tn, fp, fn, tp = confusion_matrix(
#             y_test,
#             predictions,
#         ).ravel()

#         actual_fpr = fp / (fp + tn)

#         results.append(
#             {
#                 "model": model_name,
#                 "target_fpr": target_fpr,
#                 "threshold": threshold,
#                 "actual_fpr": actual_fpr,
#                 "precision": precision,
#                 "recall": recall,
#                 "f1": f1,
#             }
#         )

#     return results


# def main():
#     print("=" * 70)
#     print("AegisSOC — Isolation Forest vs One-Class SVM")
#     print("Threshold Comparison")
#     print("=" * 70)

#     train_df, test_df = load_dataset(
#         DATA_TRAIN,
#         DATA_TEST,
#     )

#     X_train = train_df[
#         CATEGORICAL_FEATURES + NUMERIC_FEATURES
#     ]

#     X_test = test_df[
#         CATEGORICAL_FEATURES + NUMERIC_FEATURES
#     ]

#     y_train = train_df["is_anomaly"].values
#     y_test = test_df["is_anomaly"].values

#     # Normal-only training.
#     normal_mask = y_train == 0
#     X_normal = X_train.loc[normal_mask]

#     X_fit, X_calibration = train_test_split(
#         X_normal,
#         test_size=0.25,
#         random_state=42,
#     )

#     # Same preprocessing for both models.
#     preprocessor = build_preprocessor()

#     X_fit_transformed = preprocessor.fit_transform(
#         X_fit
#     )

#     X_calibration_transformed = (
#         preprocessor.transform(X_calibration)
#     )

#     X_test_transformed = (
#         preprocessor.transform(X_test)
#     )

#     models = [
#         (
#             "Isolation Forest",
#             IsolationForest(
#                 n_estimators=300,
#                 max_samples="auto",
#                 contamination=0.01,
#                 max_features=1.0,
#                 bootstrap=False,
#                 random_state=42,
#                 n_jobs=-1,
#             ),
#         ),
#         (
#             "One-Class SVM",
#             OneClassSVM(
#                 kernel="rbf",
#                 gamma="scale",
#                 nu=0.01,
#             ),
#         ),
#     ]

#     all_results = []

#     for model_name, model in models:
#         print("\n" + "-" * 70)
#         print(model_name)

#         results = evaluate_thresholds(
#             model_name,
#             model,
#             X_fit_transformed,
#             X_calibration_transformed,
#             X_test_transformed,
#             y_test,
#         )

#         all_results.extend(results)

#         print(
#             "\nTarget FPR | Actual FPR | "
#             "Precision | Recall | F1"
#         )
#         print("-" * 60)

#         for result in results:
#             print(
#                 f"{result['target_fpr']:.0%}"
#                 f"{'':8s}"
#                 f"| {result['actual_fpr']:.3f}"
#                 f"{'':5s}"
#                 f"| {result['precision']:.4f}"
#                 f"      | {result['recall']:.4f}"
#                 f" | {result['f1']:.4f}"
#             )

#     print("\n" + "=" * 70)
#     print("BEST F1 BY MODEL")
#     print("=" * 70)

#     for model_name in [
#         "Isolation Forest",
#         "One-Class SVM",
#     ]:
#         model_results = [
#             r
#             for r in all_results
#             if r["model"] == model_name
#         ]

#         best = max(
#             model_results,
#             key=lambda r: r["f1"],
#         )

#         print(
#             f"\n{model_name}"
#         )
#         print(
#             f"Target FPR: {best['target_fpr']:.0%}"
#         )
#         print(
#             f"Actual FPR: {best['actual_fpr']:.4f}"
#         )
#         print(
#             f"Precision:  {best['precision']:.4f}"
#         )
#         print(
#             f"Recall:     {best['recall']:.4f}"
#         )
#         print(
#             f"F1:         {best['f1']:.4f}"
#         )
#         print(
#             f"Threshold:  {best['threshold']:.6f}"
#         )

#     output = Path(
#         "reports/threshold_comparison.json"
#     )

#     with open(output, "w") as f:
#         json.dump(
#             all_results,
#             f,
#             indent=2,
#         )

#     print(
#         f"\nSaved to: {output}"
#     )


# if __name__ == "__main__":
#     main()
"""
threshold_family_comparison.py

Final Stage 3 check:
Compare Isolation Forest vs One-Class SVM at selected operating
thresholds, including attack-family and unseen-attack recall.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

from preprocessing import build_preprocessor, load_dataset


DATA_DIR = Path(__file__).resolve().parents[1] / "data"

RANDOM_STATE = 42

# We only care about these operating points for the final decision.
TARGET_FPRS = [0.05, 0.07, 0.10]
def evaluate_model(
    model_name,
    model,
    X_train,
    X_cal,
    X_test,
    test_df,
    y_test,
    train_df,
):
    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Fit model on NORMAL training data only
    # ---------------------------------------------------------
    start = time.time()
    model.fit(X_train)
    training_time = time.time() - start

    # ---------------------------------------------------------
    # 2. Get anomaly scores
    # Lower decision_function = more anomalous
    # ---------------------------------------------------------
    calibration_scores = model.decision_function(X_cal)
    test_scores = model.decision_function(X_test)

    # ---------------------------------------------------------
    # 3. Find attack types that appeared in training
    # ---------------------------------------------------------
    train_labels = set(train_df["label"].unique())

    # ---------------------------------------------------------
    # 4. Evaluate selected operating points
    # ---------------------------------------------------------
    for target_fpr in TARGET_FPRS:

        # Threshold is calibrated using NORMAL calibration data only
        threshold = np.quantile(
            calibration_scores,
            target_fpr
        )

        # Lower score than threshold = anomaly
        predictions = test_scores < threshold

        # -----------------------------------------------------
        # Overall metrics
        # -----------------------------------------------------
        cm = confusion_matrix(y_test, predictions)

        tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn) if (fp + tn) else 0

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0
        )

        # -----------------------------------------------------
        # Attack-family recall
        # EXACT SAME LOGIC AS train_model.py
        # -----------------------------------------------------
        recall_by_category = {}

        for cat in ["dos", "probe", "r2l", "u2r"]:

            mask = test_df["attack_category"] == cat

            if mask.sum() > 0:
                recall_by_category[cat] = float(
                    predictions[mask.values].mean()
                )

        # -----------------------------------------------------
        # Novel / unseen attack recall
        # EXACT SAME LOGIC AS train_model.py
        # -----------------------------------------------------
        novel_mask = (
            (test_df["is_anomaly"] == 1)
            & (~test_df["label"].isin(train_labels))
        ).values

        if novel_mask.sum() > 0:
            recall_novel = float(
                predictions[novel_mask].mean()
            )
        else:
            recall_novel = None

        # -----------------------------------------------------
        # Print results
        # -----------------------------------------------------
        print(
            f"\nTarget FPR: {target_fpr * 100:.0f}%"
        )

        print(
            f"Threshold={threshold:.6f} | "
            f"Actual FPR={fpr:.4f} | "
            f"Precision={precision:.4f} | "
            f"Recall={recall:.4f} | "
            f"F1={f1:.4f}"
        )

        print(
            f"DoS={recall_by_category.get('dos', 0):.4f} | "
            f"Probe={recall_by_category.get('probe', 0):.4f} | "
            f"R2L={recall_by_category.get('r2l', 0):.4f} | "
            f"U2R={recall_by_category.get('u2r', 0):.4f} | "
            f"Unseen={recall_novel:.4f}"
            if recall_novel is not None
            else
            f"DoS={recall_by_category.get('dos', 0):.4f} | "
            f"Probe={recall_by_category.get('probe', 0):.4f} | "
            f"R2L={recall_by_category.get('r2l', 0):.4f} | "
            f"U2R={recall_by_category.get('u2r', 0):.4f} | "
            f"Unseen=n/a"
        )

    print(f"\nTraining time: {training_time:.2f}s")


def main():

    print("Loading NSL-KDD...")

    train_df, test_df = load_dataset(
        str(DATA_DIR / "KDDTrain+.txt"),
        str(DATA_DIR / "KDDTest+.txt")
    )


    # ---------------------------------------------------------
    # Separate normal training traffic
    # ---------------------------------------------------------
    normal_train_df = train_df[
        train_df["label"] == "normal"
    ].copy()

    # 75% model fitting / 25% threshold calibration
    normal_fit_df, normal_cal_df = train_test_split(
        normal_train_df,
        test_size=0.25,
        random_state=RANDOM_STATE
    )

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------
    preprocessor = build_preprocessor()

    X_train = preprocessor.fit_transform(
        normal_fit_df
    )

    X_cal = preprocessor.transform(
        normal_cal_df
    )

    X_test = preprocessor.transform(
        test_df
    )

    y_test = (
        test_df["label"] != "normal"
    ).astype(int).values

    print(f"Train shape: {X_train.shape}")
    print(f"Calibration shape: {X_cal.shape}")
    print(f"Test shape: {X_test.shape}")

    # ---------------------------------------------------------
    # Isolation Forest
    # ---------------------------------------------------------
    isolation_forest = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination=0.01,
        max_features=1.0,
        bootstrap=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    evaluate_model(
        "ISOLATION FOREST",
        isolation_forest,
        X_train,
        X_cal,
        X_test,
        test_df,
        y_test,
        train_df,
    )

    # ---------------------------------------------------------
    # One-Class SVM
    # ---------------------------------------------------------
    svm = OneClassSVM(
        kernel="rbf",
        gamma="scale",
        nu=0.01,
    )

    evaluate_model(
        "ONE-CLASS SVM",
        svm,
        X_train,
        X_cal,
        X_test,
        test_df,
        y_test,
        train_df,
    )


if __name__ == "__main__":
    main()
