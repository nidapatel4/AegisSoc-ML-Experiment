import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
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
    ENGINEERED_FEATURES,
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
)


DATA_TRAIN = "data/KDDTrain+.txt"
DATA_TEST = "data/KDDTest+.txt"

BASELINE_FEATURES = [
    f for f in ALL_FEATURES if f not in ENGINEERED_FEATURES
]


def evaluate(feature_set, name):
    train_df, test_df = load_dataset(DATA_TRAIN, DATA_TEST)

    # Keep only the selected features.
    X_train = train_df[feature_set]
    X_test = test_df[feature_set]

    y_train = train_df["is_anomaly"].values
    y_test = test_df["is_anomaly"].values

    # Normal-only training.
    normal_mask = y_train == 0
    X_normal = X_train.loc[normal_mask]

    # Same 75/25 normal fit/calibration split used by baseline.
    X_fit, X_calibration = train_test_split(
        X_normal,
        test_size=0.25,
        random_state=42,
    )

    # Same preprocessing.
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    categorical_features = [
        f for f in CATEGORICAL_FEATURES if f in feature_set
    ]

    numeric_features = [
        f for f in feature_set if f not in categorical_features
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
            (
                "num",
                StandardScaler(),
                numeric_features,
            ),
        ]
    )

    X_fit_transformed = preprocessor.fit_transform(X_fit)
    X_calibration_transformed = preprocessor.transform(X_calibration)
    X_test_transformed = preprocessor.transform(X_test)

    # Same Isolation Forest configuration.
    model = IsolationForest(
        n_estimators=300,
        contamination=0.01,
        random_state=42,
        n_jobs=-1,
    )

    start = time.time()
    model.fit(X_fit_transformed)
    train_time = time.time() - start

    # Calibrate threshold at 5% FPR using ONLY held-out normal traffic.
    calibration_scores = model.decision_function(
        X_calibration_transformed
    )
    threshold = float(
        np.quantile(calibration_scores, 0.05)
    )

    test_scores = model.decision_function(
        X_test_transformed
    )
    predictions = (test_scores < threshold).astype(int)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )
    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )
    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )
    roc_auc = roc_auc_score(
        y_test,
        -test_scores,
    )
    pr_auc = average_precision_score(
        y_test,
        -test_scores,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
    ).ravel()

    fpr = fp / (fp + tn)

    # Attack-family recall.
    family_recalls = {}

    for family in ["dos", "probe", "r2l", "u2r"]:
        mask = test_df["attack_category"].values == family

        if mask.sum() > 0:
            family_recalls[family] = float(
                recall_score(
                    y_test[mask],
                    predictions[mask],
                    zero_division=0,
                )
            )

    # Recall on unseen attack types.
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
        "name": name,
        "features": len(feature_set),
        "encoded_dimensions": X_fit_transformed.shape[1],
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "family_recall": family_recalls,
        "unseen_recall": unseen_recall,
        "train_time_seconds": train_time,
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
    }


def main():
    experiments = [
        ("baseline", BASELINE_FEATURES),
        (
            "all_5_engineered",
            ALL_FEATURES,
        ),
    ]

    # One engineered feature removed at a time.
    for feature in ENGINEERED_FEATURES:
        features = [
            f for f in ALL_FEATURES
            if f != feature
        ]

        experiments.append(
            (f"without_{feature}", features)
        )

    results = []

    print("=" * 70)
    print("AegisSOC — Feature Ablation Experiment")
    print("=" * 70)

    for name, features in experiments:
        print(f"\nRunning: {name}")
        print(f"Features: {len(features)}")

        result = evaluate(features, name)
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

    results_dir = Path("reports")
    results_dir.mkdir(exist_ok=True)

    output = results_dir / "feature_ablation.json"

    with open(output, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    ranked = sorted(
        results,
        key=lambda x: x["f1"],
        reverse=True,
    )

    for result in ranked:
        print(
            f"{result['name']:35s} "
            f"F1={result['f1']:.4f} "
            f"Recall={result['recall']:.4f} "
            f"PR-AUC={result['pr_auc']:.4f}"
        )

    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()

    