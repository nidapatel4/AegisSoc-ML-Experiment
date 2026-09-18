from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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

DATA_DIR = BASE_DIR / "data/MachineLearningCVE"
MODEL_DIR = BASE_DIR / "models/CIC"
REPORT_PATH = BASE_DIR / "reports/CIC/cic2017_feature_engineering.json"


TRAIN_FILE = "Monday-WorkingHours.pcap_ISCX.csv"
CAL_FILE = "Tuesday-WorkingHours.pcap_ISCX.csv"

TEST_FILES = [
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


# The 8 constant features discovered during the CIC audit.
CONSTANT_COLUMNS = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]


def load_csv(filename):
    path = DATA_DIR / filename

    df = pd.read_csv(path)

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    return df


def add_engineered_features(df):

    df = df.copy()

    # Avoid division by zero.
    duration = df["Flow Duration"].replace(0, np.nan)

    fwd_packets = df["Total Fwd Packets"]
    bwd_packets = df["Total Backward Packets"]

    fwd_bytes = df["Total Length of Fwd Packets"]
    bwd_bytes = df["Total Length of Bwd Packets"]

    # 1. Total packets in the flow.
    df["engineered_packet_count"] = (
        fwd_packets + bwd_packets
    )

    # 2. Total bytes in the flow.
    df["engineered_byte_count"] = (
        fwd_bytes + bwd_bytes
    )

    # CIC Flow Duration is in microseconds.
    # Convert to seconds before calculating rates.
    duration_seconds = duration / 1_000_000

    # 3. Packets per second.
    df["engineered_packets_per_second"] = (
        df["engineered_packet_count"]
        / duration_seconds
    )

    # 4. Bytes per second.
    df["engineered_bytes_per_second"] = (
        df["engineered_byte_count"]
        / duration_seconds
    )

    # 5. Forward/backward packet ratio.
    df["engineered_fwd_bwd_packet_ratio"] = (
        fwd_packets
        / bwd_packets.replace(0, np.nan)
    )

    return df


def prepare_features(df, add_engineered):

    df = df.copy()

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    if add_engineered:
        df = add_engineered_features(df)

    # Remove label.
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # Remove the known constant columns.
    existing_constants = [
        col for col in CONSTANT_COLUMNS
        if col in df.columns
    ]

    df = df.drop(
        columns=existing_constants,
        errors="ignore"
    )

    # Convert all remaining columns to numeric.
    for col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Infinity becomes NaN and is handled by the imputer.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


def build_preprocessor(feature_names):

    return Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
    ])


def evaluate_model(model, X_cal, X_test, y_test):

    cal_scores = model.decision_function(X_cal)
    test_scores = model.decision_function(X_test)

    # 5% target FPR using Tuesday BENIGN calibration data.
    threshold = np.percentile(
        cal_scores,
        5
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
        else 0
    )

    return {
        "threshold": float(threshold),
        "accuracy": float(
            accuracy_score(y_test, y_pred)
        ),
        "precision": float(
            precision_score(
                y_test,
                y_pred,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                y_pred,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                y_pred,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                -test_scores
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_test,
                -test_scores
            )
        ),
        "actual_test_fpr": float(
            actual_fpr
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def main():

    print("=" * 70)
    print("CIC-IDS2017 TARGETED FEATURE ENGINEERING")
    print("=" * 70)

    print()
    print("Loading training data...")
    train_df = load_csv(TRAIN_FILE)

    print("Loading calibration data...")
    cal_df = load_csv(CAL_FILE)

    print("Loading test data...")

    test_parts = []

    for filename in TEST_FILES:
        print(f"  {filename}")
        test_parts.append(
            load_csv(filename)
        )

    test_df = pd.concat(
        test_parts,
        ignore_index=True
    )

    y_test = (
        test_df["Label"]
        .astype(str)
        .str.strip()
        .str.upper()
        .ne("BENIGN")
        .astype(int)
        .to_numpy()
    )

    results = {}

    for experiment_name, use_engineered in [
        ("original_70_features", False),
        ("original_plus_5_engineered", True),
    ]:

        print()
        print("=" * 70)
        print(experiment_name)
        print("=" * 70)

        X_train_df = prepare_features(
            train_df,
            use_engineered
        )

        X_cal_df = prepare_features(
            cal_df,
            use_engineered
        )

        X_test_df = prepare_features(
            test_df,
            use_engineered
        )

        print(
            f"Feature count: {X_train_df.shape[1]}"
        )

        if list(X_train_df.columns) != list(
            X_cal_df.columns
        ):
            raise ValueError(
                "Training and calibration feature mismatch."
            )

        if list(X_train_df.columns) != list(
            X_test_df.columns
        ):
            raise ValueError(
                "Training and test feature mismatch."
            )

        feature_names = list(
            X_train_df.columns
        )

        preprocessor = build_preprocessor(
            feature_names
        )

        print("Fitting preprocessing on training data only...")

        X_train = preprocessor.fit_transform(
            X_train_df
        )

        X_cal = preprocessor.transform(
            X_cal_df
        )

        X_test = preprocessor.transform(
            X_test_df
        )

        print(
            f"Processed shape: {X_train.shape}"
        )

        model = IsolationForest(
            n_estimators=800,
            max_samples=1.0,
            max_features=0.7,
            contamination=0.01,
            bootstrap=False,
            random_state=42,
            n_jobs=-1,
        )

        print("Training Isolation Forest...")

        model.fit(X_train)

        metrics = evaluate_model(
            model,
            X_cal,
            X_test,
            y_test
        )

        results[experiment_name] = {
            "feature_count": len(feature_names),
            "features": feature_names,
            "metrics": metrics,
        }

        print()
        print(
            f"Threshold:     "
            f"{metrics['threshold']:.6f}"
        )
        print(
            f"Actual FPR:    "
            f"{metrics['actual_test_fpr'] * 100:.2f}%"
        )
        print(
            f"Precision:     "
            f"{metrics['precision']:.4f}"
        )
        print(
            f"Recall:        "
            f"{metrics['recall']:.4f}"
        )
        print(
            f"F1:            "
            f"{metrics['f1']:.4f}"
        )
        print(
            f"ROC-AUC:       "
            f"{metrics['roc_auc']:.4f}"
        )
        print(
            f"PR-AUC:        "
            f"{metrics['pr_auc']:.4f}"
        )

    # Calculate changes.
    baseline = results[
        "original_70_features"
    ]["metrics"]

    engineered = results[
        "original_plus_5_engineered"
    ]["metrics"]

    comparison = {
        "f1_change": (
            engineered["f1"]
            - baseline["f1"]
        ),
        "recall_change": (
            engineered["recall"]
            - baseline["recall"]
        ),
        "precision_change": (
            engineered["precision"]
            - baseline["precision"]
        ),
        "roc_auc_change": (
            engineered["roc_auc"]
            - baseline["roc_auc"]
        ),
        "pr_auc_change": (
            engineered["pr_auc"]
            - baseline["pr_auc"]
        ),
        "actual_fpr_change": (
            engineered["actual_test_fpr"]
            - baseline["actual_test_fpr"]
        ),
    }

    report = {
        "experiment": (
            "CIC-IDS2017 targeted feature engineering"
        ),

        "reason": (
            "Attack-family analysis showed very low "
            "recall for PortScan, Bot and WebAttack. "
            "Five traffic-volume, rate and direction "
            "features were tested."
        ),

        "engineered_features": [
            "engineered_packet_count",
            "engineered_byte_count",
            "engineered_packets_per_second",
            "engineered_bytes_per_second",
            "engineered_fwd_bwd_packet_ratio",
        ],

        "results": results,

        "comparison": comparison,
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
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Baseline F1:   "
        f"{baseline['f1']:.4f}"
    )

    print(
        f"Engineered F1: "
        f"{engineered['f1']:.4f}"
    )

    print(
        f"F1 change:     "
        f"{comparison['f1_change']:+.4f}"
    )

    print()
    print(
        f"Report saved to:\n{REPORT_PATH}"
    )


if __name__ == "__main__":
    main()