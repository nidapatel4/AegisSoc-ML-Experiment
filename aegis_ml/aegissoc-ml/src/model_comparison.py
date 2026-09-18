import json
import time
from pathlib import Path

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from preprocessing import (
    load_dataset,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)


DATA_TRAIN = "data/KDDTrain+.txt"
DATA_TEST = "data/KDDTest+.txt"

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES,
            ),
        ]
    )


def evaluate_model(name, model, X_fit, X_calibration, X_test, y_test, test_df):
    start = time.time()
    model.fit(X_fit)
    train_time = time.time() - start

    # Convert model output so that LOWER values mean "more anomalous".
    if name == "LOF":
        calibration_scores = -model.decision_function(X_calibration)
        test_scores = -model.decision_function(X_test)
    else:
        calibration_scores = model.decision_function(X_calibration)
        test_scores = model.decision_function(X_test)

    # Calibrate threshold using ONLY normal calibration data.
    if name == "LOF":
        threshold = float(np.quantile(calibration_scores, 0.95))
        predictions = (test_scores > threshold).astype(int)
        anomaly_scores = test_scores
    else:
        threshold = float(np.quantile(calibration_scores, 0.05))
        predictions = (test_scores < threshold).astype(int)
        anomaly_scores = -test_scores

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(
        y_test, anomaly_scores
    )
    pr_auc = average_precision_score(
        y_test, anomaly_scores
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test, predictions
    ).ravel()

    fpr = fp / (fp + tn)

    # Attack-family recall.
    family_recalls = {}

    for family in ["dos", "probe", "r2l", "u2r"]:
        mask = (
            test_df["attack_category"].values
            == family
        )

        if mask.sum() > 0:
            family_recalls[family] = float(
                recall_score(
                    y_test[mask],
                    predictions[mask],
                    zero_division=0,
                )
            )

    # Unseen attack recall.
    train_df, _ = load_dataset(
        DATA_TRAIN,
        DATA_TEST,
    )

    train_attack_labels = set(
        train_df.loc[
            train_df["is_anomaly"] == 1,
            "label",
        ]
    )

    unseen_mask = (
        (test_df["is_anomaly"].values == 1)
        & ~test_df["label"].isin(
            train_attack_labels
        ).values
    )

    unseen_recall = None

    if unseen_mask.sum() > 0:
        unseen_recall = float(
            recall_score(
                y_test[unseen_mask],
                predictions[unseen_mask],
                zero_division=0,
            )
        )

    return {
        "model": name,
        "threshold": threshold,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "fpr": float(fpr),
        "family_recall": family_recalls,
        "unseen_recall": unseen_recall,
        "train_time_seconds": float(train_time),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
    }


def main():
    print("=" * 70)
    print("AegisSOC — Model Comparison Experiment")
    print("=" * 70)

    # Load original 41-feature dataset.
    train_df, test_df = load_dataset(
        DATA_TRAIN,
        DATA_TEST,
    )

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]

    y_train = train_df["is_anomaly"].values
    y_test = test_df["is_anomaly"].values

    # Train only on normal traffic.
    normal_mask = y_train == 0
    X_normal = X_train.loc[normal_mask]

    # Same 75/25 split as the baseline.
    X_fit, X_calibration = train_test_split(
        X_normal,
        test_size=0.25,
        random_state=42,
    )

    # Same preprocessing for every model.
    preprocessor = build_preprocessor()

    X_fit_transformed = preprocessor.fit_transform(X_fit)
    X_calibration_transformed = preprocessor.transform(
        X_calibration
    )
    X_test_transformed = preprocessor.transform(
        X_test
    )

    print(
        f"\nTraining samples: {X_fit.shape[0]}"
    )
    print(
        f"Calibration samples: {X_calibration.shape[0]}"
    )
    print(
        f"Test samples: {X_test.shape[0]}"
    )
    print(
        f"Encoded dimensions: "
        f"{X_fit_transformed.shape[1]}"
    )

    # Three anomaly-detection algorithms.
    models = [
        (
            "Isolation Forest",
            IsolationForest(
                n_estimators=300,
                max_samples="auto",
                contamination=0.01,
                max_features=1.0,
                bootstrap=False,
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "One-Class SVM",
            OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=0.01,
            ),
        ),
        (
            "LOF",
            LocalOutlierFactor(
                n_neighbors=20,
                algorithm="auto",
                metric="minkowski",
                contamination=0.01,
                novelty=True,
                n_jobs=-1,
            ),
        ),
    ]

    results = []

    for name, model in models:
        print("\n" + "-" * 70)
        print(f"Running: {name}")

        result = evaluate_model(
            name,
            model,
            X_fit_transformed,
            X_calibration_transformed,
            X_test_transformed,
            y_test,
            test_df,
        )

        results.append(result)

        print(
            f"F1={result['f1']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"Precision={result['precision']:.4f} | "
            f"PR-AUC={result['pr_auc']:.4f} | "
            f"ROC-AUC={result['roc_auc']:.4f} | "
            f"FPR={result['fpr']:.4f}"
        )

        print(
            f"DoS={result['family_recall'].get('dos', 0):.4f} | "
            f"Probe={result['family_recall'].get('probe', 0):.4f} | "
            f"R2L={result['family_recall'].get('r2l', 0):.4f} | "
            f"U2R={result['family_recall'].get('u2r', 0):.4f} | "
            f"Unseen={result['unseen_recall']:.4f}"
        )

        print(
            f"Training time="
            f"{result['train_time_seconds']:.2f}s"
        )

    # Rank by F1.
    ranked = sorted(
        results,
        key=lambda x: x["f1"],
        reverse=True,
    )

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    for result in ranked:
        print(
            f"{result['model']:20s} "
            f"F1={result['f1']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"Precision={result['precision']:.4f} | "
            f"PR-AUC={result['pr_auc']:.4f} | "
            f"FPR={result['fpr']:.4f}"
        )

    results_dir = Path("reports")
    results_dir.mkdir(exist_ok=True)

    output = (
        results_dir
        / "model_comparison.json"
    )

    with open(output, "w") as f:
        json.dump(
            results,
            f,
            indent=2,
        )

    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
    