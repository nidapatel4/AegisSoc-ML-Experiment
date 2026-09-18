"""
CIC-IDS2017 Supervised Random Forest Baseline

Purpose:
    Test whether a supervised tree model can materially outperform
    the current Isolation Forest approach on CIC-IDS2017.

Dataset split:

    SUPERVISED TRAINING SOURCE:
        Monday BENIGN
        +
        Tuesday BENIGN + ATTACK

    TRAIN / VALIDATION:
        80/20 stratified split of the combined Monday + Tuesday data

    TEST:
        Wednesday + Thursday + Friday
        Completely untouched until final evaluation

Important:
    This script creates a NEW experiment.
    It does not modify or overwrite the existing Isolation Forest
    experiments.
"""

from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "MachineLearningCVE"

MODEL_DIR = BASE_DIR / "models" / "CIC"
REPORT_DIR = BASE_DIR / "reports" / "CIC"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASET FILES
# ============================================================

TRAIN_FILE = DATA_DIR / "Monday-WorkingHours.pcap_ISCX.csv"

VALIDATION_FILE = DATA_DIR / "Tuesday-WorkingHours.pcap_ISCX.csv"

TEST_FILES = [
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


# ============================================================
# MODEL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 300

CLASS_WEIGHT = "balanced_subsample"

N_JOBS = -1


# ============================================================
# CONSTANT FEATURES REMOVED IN OUR CIC PIPELINE
# ============================================================

CONSTANT_FEATURES = [
    "Bwd PSH Flags",
    "Bwd URG Flags",
    "Fwd Avg Bytes/Bulk",
    "Fwd Avg Packets/Bulk",
    "Fwd Avg Bulk Rate",
    "Bwd Avg Bytes/Bulk",
    "Bwd Avg Packets/Bulk",
    "Bwd Avg Bulk Rate",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_columns(df):
    """
    Remove leading/trailing whitespace from column names.
    """

    df.columns = df.columns.astype(str).str.strip()

    return df


def clean_labels(df):
    """
    Convert CIC-IDS2017 labels into:

        0 = BENIGN
        1 = ATTACK
    """

    if "Label" not in df.columns:
        raise ValueError(
            "Label column not found. Available columns:\n"
            + "\n".join(df.columns)
        )

    labels = (
        df["Label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    y = (labels != "BENIGN").astype(np.int8)

    return y


def prepare_features(df, feature_columns=None):
    """
    Prepare CIC features.

    Steps:
        1. Remove Label
        2. Remove constant features
        3. Convert everything to numeric
        4. Replace infinities with NaN

    No fitting happens here.
    """

    df = df.copy()

    # Remove target label from feature data.
    if "Label" in df.columns:
        df = df.drop(columns=["Label"])

    # Remove the same constant features used in our
    # existing CIC-IDS2017 preprocessing.
    existing_constants = [
        col
        for col in CONSTANT_FEATURES
        if col in df.columns
    ]

    if existing_constants:
        df = df.drop(
            columns=existing_constants
        )

    # If feature columns were supplied, enforce exactly
    # the same feature ordering.
    if feature_columns is not None:

        missing = [
            col
            for col in feature_columns
            if col not in df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing expected features: {missing}"
            )

        df = df[feature_columns]

    # Convert all remaining columns to numeric.
    for col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Replace +/- infinity with NaN.
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


def calculate_metrics(y_true, y_pred, y_score):
    """
    Calculate standard binary classification metrics.
    """

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    normal_count = tn + fp

    actual_fpr = (
        fp / normal_count
        if normal_count > 0
        else 0.0
    )

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred
            )
        ),

        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                y_true,
                y_score
            )
        ),

        "pr_auc": float(
            average_precision_score(
                y_true,
                y_score
            )
        ),

        "actual_fpr": float(
            actual_fpr
        ),

        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
        }
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CIC-IDS2017 SUPERVISED RANDOM FOREST BASELINE")
    print("=" * 70)

    start_time = time.time()

    # ========================================================
    # LOAD SUPERVISED TRAINING SOURCE
    # ========================================================

    print("\nLoading supervised training data...")

    # --------------------------------------------------------
    # MONDAY
    # --------------------------------------------------------

    print(f"\n  {TRAIN_FILE.name}")

    monday_df = pd.read_csv(
        TRAIN_FILE,
        low_memory=False
    )

    monday_df = clean_columns(
        monday_df
    )

    monday_labels = clean_labels(
        monday_df
    )

    print(
        f"  Monday rows: "
        f"{len(monday_df):,}"
    )

    print(
        f"  Monday normal: "
        f"{int((monday_labels == 0).sum()):,}"
    )

    print(
        f"  Monday attacks: "
        f"{int((monday_labels == 1).sum()):,}"
    )

    # --------------------------------------------------------
    # TUESDAY
    # --------------------------------------------------------

    print(f"\n  {VALIDATION_FILE.name}")

    tuesday_df = pd.read_csv(
        VALIDATION_FILE,
        low_memory=False
    )

    tuesday_df = clean_columns(
        tuesday_df
    )

    tuesday_labels = clean_labels(
        tuesday_df
    )

    print(
        f"  Tuesday rows: "
        f"{len(tuesday_df):,}"
    )

    print(
        f"  Tuesday normal: "
        f"{int((tuesday_labels == 0).sum()):,}"
    )

    print(
        f"  Tuesday attacks: "
        f"{int((tuesday_labels == 1).sum()):,}"
    )

    # --------------------------------------------------------
    # COMBINE MONDAY + TUESDAY
    # --------------------------------------------------------

    print("\nCombining Monday + Tuesday...")

    supervised_df = pd.concat(
        [
            monday_df,
            tuesday_df
        ],
        ignore_index=True
    )

    supervised_labels = pd.concat(
        [
            pd.Series(monday_labels),
            pd.Series(tuesday_labels)
        ],
        ignore_index=True
    ).to_numpy(
        dtype=np.int8
    )

    print(
        f"  Combined rows: "
        f"{len(supervised_df):,}"
    )

    print(
        f"  Combined normal: "
        f"{int((supervised_labels == 0).sum()):,}"
    )

    print(
        f"  Combined attacks: "
        f"{int((supervised_labels == 1).sum()):,}"
    )

    # ========================================================
    # STRATIFIED TRAIN / VALIDATION SPLIT
    # ========================================================

    print("\nCreating stratified train/validation split...")

    (
        train_df,
        validation_df,
        y_train,
        y_validation
    ) = train_test_split(
        supervised_df,
        supervised_labels,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=supervised_labels,
    )

    train_df = train_df.reset_index(
        drop=True
    )

    validation_df = validation_df.reset_index(
        drop=True
    )

    y_train = np.asarray(
        y_train,
        dtype=np.int8
    )

    y_validation = np.asarray(
        y_validation,
        dtype=np.int8
    )

    print("\nSupervised split:")

    print(
        f"  Training rows: "
        f"{len(train_df):,}"
    )

    print(
        f"  Training normal: "
        f"{int((y_train == 0).sum()):,}"
    )

    print(
        f"  Training attacks: "
        f"{int((y_train == 1).sum()):,}"
    )

    print(
        f"  Validation rows: "
        f"{len(validation_df):,}"
    )

    print(
        f"  Validation normal: "
        f"{int((y_validation == 0).sum()):,}"
    )

    print(
        f"  Validation attacks: "
        f"{int((y_validation == 1).sum()):,}"
    )

    # ========================================================
    # LOAD UNTOUCHED TEST DATA
    # ========================================================

    print("\nLoading untouched test data...")

    test_frames = []
    test_labels = []

    for filename in TEST_FILES:

        print(f"  {filename}")

        path = DATA_DIR / filename

        df = pd.read_csv(
            path,
            low_memory=False
        )

        df = clean_columns(
            df
        )

        y = clean_labels(
            df
        )

        test_frames.append(
            df
        )

        test_labels.append(
            y
        )

    test_df = pd.concat(
        test_frames,
        ignore_index=True
    )

    y_test = pd.concat(
        [
            pd.Series(y)
            for y in test_labels
        ],
        ignore_index=True
    ).to_numpy(
        dtype=np.int8
    )

    print(
        f"\n  Test rows: "
        f"{len(test_df):,}"
    )

    print(
        f"  Test normal: "
        f"{int((y_test == 0).sum()):,}"
    )

    print(
        f"  Test attacks: "
        f"{int((y_test == 1).sum()):,}"
    )

    # ========================================================
    # PREPARE FEATURES
    # ========================================================

    print("\nPreparing features...")

    X_train_df = prepare_features(
        train_df
    )

    feature_columns = list(
        X_train_df.columns
    )

    X_validation_df = prepare_features(
        validation_df,
        feature_columns=feature_columns
    )

    X_test_df = prepare_features(
        test_df,
        feature_columns=feature_columns
    )

    print(
        f"  Feature count: "
        f"{len(feature_columns)}"
    )

    print(
        f"  Training shape: "
        f"{X_train_df.shape}"
    )

    print(
        f"  Validation shape: "
        f"{X_validation_df.shape}"
    )

    print(
        f"  Test shape: "
        f"{X_test_df.shape}"
    )

    # ========================================================
    # PREPROCESSING
    # ========================================================

    print(
        "\nFitting preprocessing on training data only..."
    )

    numeric_features = feature_columns

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",

                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            )
                        ),

                        (
                            "scaler",
                            StandardScaler()
                        ),
                    ]
                ),

                numeric_features,
            )
        ],

        remainder="drop",
    )

    # Fit ONLY on training data.
    X_train = preprocessor.fit_transform(
        X_train_df
    )

    # Transform validation/test using
    # the training-fitted preprocessing.
    X_validation = preprocessor.transform(
        X_validation_df
    )

    X_test = preprocessor.transform(
        X_test_df
    )

    print(
        f"  Processed training shape: "
        f"{X_train.shape}"
    )

    print(
        f"  Processed validation shape: "
        f"{X_validation.shape}"
    )

    print(
        f"  Processed test shape: "
        f"{X_test.shape}"
    )

    # ========================================================
    # TRAIN RANDOM FOREST
    # ========================================================

    print("\n" + "=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    print(
        f"n_estimators: "
        f"{N_ESTIMATORS}"
    )

    print(
        f"class_weight: "
        f"{CLASS_WEIGHT}"
    )

    print(
        f"random_state: "
        f"{RANDOM_STATE}"
    )

    print(
        f"n_jobs: "
        f"{N_JOBS}"
    )

    print("\nTraining...")

    model_start = time.time()

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        class_weight=CLASS_WEIGHT,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        max_features="sqrt",
        bootstrap=True,
    )

    model.fit(
        X_train,
        y_train
    )

    model_time = (
        time.time()
        - model_start
    )

    print(
        f"Training completed in "
        f"{model_time:.2f} seconds."
    )

    print(
        f"\nLearned classes: "
        f"{model.classes_}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)

    validation_scores = model.predict_proba(
        X_validation
    )[:, 1]

    validation_predictions = (
        validation_scores >= 0.5
    ).astype(
        np.int8
    )

    validation_metrics = calculate_metrics(
        y_validation,
        validation_predictions,
        validation_scores
    )

    print(
        f"Validation F1:        "
        f"{validation_metrics['f1']:.4f}"
    )

    print(
        f"Validation Precision: "
        f"{validation_metrics['precision']:.4f}"
    )

    print(
        f"Validation Recall:    "
        f"{validation_metrics['recall']:.4f}"
    )

    print(
        f"Validation ROC-AUC:   "
        f"{validation_metrics['roc_auc']:.4f}"
    )

    print(
        f"Validation PR-AUC:    "
        f"{validation_metrics['pr_auc']:.4f}"
    )

    print(
        f"Validation FPR:       "
        f"{validation_metrics['actual_fpr'] * 100:.2f}%"
    )

    # ========================================================
    # THRESHOLD
    # ========================================================
    #
    # We initially use the standard 0.5 threshold.
    #
    # We are NOT performing another threshold sweep here.
    #
    # This experiment is primarily answering:
    #
    # "Can supervised learning substantially outperform
    # Isolation Forest on CIC-IDS2017?"
    #
    # If the supervised model is promising, a controlled
    # threshold experiment can be performed later.
    # ========================================================

    threshold = 0.5

    print(
        f"\nUsing validation threshold: "
        f"{threshold:.2f}"
    )

    # ========================================================
    # FINAL TEST EVALUATION
    # ========================================================

    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON UNTOUCHED TEST")
    print("=" * 70)

    print(
        "Evaluating Wednesday + Thursday + Friday..."
    )

    test_scores = model.predict_proba(
        X_test
    )[:, 1]

    test_predictions = (
        test_scores >= threshold
    ).astype(
        np.int8
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
        test_scores
    )

    print("\nRESULTS")
    print("-" * 70)

    print(
        f"Accuracy:       "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision:      "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall:         "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1:             "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"ROC-AUC:        "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:         "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        f"Actual FPR:     "
        f"{test_metrics['actual_fpr'] * 100:.2f}%"
    )

    print("\nConfusion Matrix:")

    cm = test_metrics[
        "confusion_matrix"
    ]

    print(
        f"  TN = {cm['TN']:,}"
    )

    print(
        f"  FP = {cm['FP']:,}"
    )

    print(
        f"  FN = {cm['FN']:,}"
    )

    print(
        f"  TP = {cm['TP']:,}"
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    print(
        "\nCalculating feature importance..."
    )

    importances = (
        model.feature_importances_
    )

    feature_importance = sorted(
        zip(
            feature_columns,
            importances
        ),
        key=lambda x: x[1],
        reverse=True
    )

    print("\nTop 20 features:")

    for feature, importance in (
        feature_importance[:20]
    ):

        print(
            f"  {feature:<40} "
            f"{importance:.6f}"
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    model_path = (
        MODEL_DIR
        / "cic2017_random_forest.joblib"
    )

    preprocessor_path = (
        MODEL_DIR
        / "cic2017_random_forest_preprocessor.joblib"
    )

    print("\nSaving model...")

    joblib.dump(
        model,
        model_path
    )

    joblib.dump(
        preprocessor,
        preprocessor_path
    )

    # ========================================================
    # SAVE REPORT
    # ========================================================

    report = {

        "experiment":
            "CIC-IDS2017 supervised Random Forest baseline",

        "dataset":
            "CIC-IDS2017",

        "dataset_source":
            "Canadian Institute for Cybersecurity",

        "random_state":
            RANDOM_STATE,

        "model": {

            "type":
                "RandomForestClassifier",

            "n_estimators":
                N_ESTIMATORS,

            "max_features":
                "sqrt",

            "class_weight":
                CLASS_WEIGHT,

            "bootstrap":
                True,

            "n_jobs":
                N_JOBS,
        },

        "split": {

            "training_source":
                "Monday BENIGN + Tuesday all traffic",

            "training":
                "80% stratified split of combined Monday/Tuesday data",

            "validation":
                "20% stratified split of combined Monday/Tuesday data",

            "test":
                "Wednesday + Thursday + Friday all traffic",
        },

        "feature_count":
            len(feature_columns),

        "features":
            feature_columns,

        "threshold":
            threshold,

        "training_rows":
            int(len(y_train)),

        "training_normal_rows":
            int((y_train == 0).sum()),

        "training_attack_rows":
            int((y_train == 1).sum()),

        "validation_rows":
            int(len(y_validation)),

        "validation_normal_rows":
            int((y_validation == 0).sum()),

        "validation_attack_rows":
            int((y_validation == 1).sum()),

        "test_rows":
            int(len(y_test)),

        "test_normal_rows":
            int((y_test == 0).sum()),

        "test_attack_rows":
            int((y_test == 1).sum()),

        "validation":
            validation_metrics,

        "test":
            test_metrics,

        "top_20_feature_importance": [

            {
                "feature":
                    feature,

                "importance":
                    float(importance)
            }

            for feature, importance
            in feature_importance[:20]
        ],

        "training_time_seconds":
            float(model_time),

        "total_runtime_seconds":
            float(
                time.time()
                - start_time
            ),

        "model_path":
            str(model_path),

        "preprocessor_path":
            str(preprocessor_path),
    }

    report_path = (
        REPORT_DIR
        / "cic2017_random_forest.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2
        )

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print("\n" + "=" * 70)
    print("COMPARISON WITH CURRENT ISOLATION FOREST")
    print("=" * 70)

    print(
        f"{'Metric':<20}"
        f"{'Isolation Forest':>20}"
        f"{'Random Forest':>20}"
    )

    print("-" * 60)

    print(
        f"{'F1':<20}"
        f"{0.5716:>20.4f}"
        f"{test_metrics['f1']:>20.4f}"
    )

    print(
        f"{'Precision':<20}"
        f"{0.7360:>20.4f}"
        f"{test_metrics['precision']:>20.4f}"
    )

    print(
        f"{'Recall':<20}"
        f"{0.4672:>20.4f}"
        f"{test_metrics['recall']:>20.4f}"
    )

    print(
        f"{'ROC-AUC':<20}"
        f"{0.8257:>20.4f}"
        f"{test_metrics['roc_auc']:>20.4f}"
    )

    print(
        f"{'PR-AUC':<20}"
        f"{0.6677:>20.4f}"
        f"{test_metrics['pr_auc']:>20.4f}"
    )

    print(
        f"{'Actual FPR':<20}"
        f"{'6.95%':>20}"
        f"{test_metrics['actual_fpr'] * 100:>19.2f}%"
    )

    f1_change = (
        test_metrics["f1"]
        - 0.5716
    )

    print("\nF1 change:")

    print(
        f"  {f1_change:+.4f}"
    )

    print("\nReport saved to:")

    print(
        report_path
    )

    print("\nModel saved to:")

    print(
        model_path
    )

    print("\nPreprocessor saved to:")

    print(
        preprocessor_path
    )

    print("\n" + "=" * 70)
    print("RANDOM FOREST EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()