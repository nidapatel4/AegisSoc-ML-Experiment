import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
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
# FILES
# ============================================================

MONDAY = "Monday-WorkingHours.pcap_ISCX.csv"
TUESDAY = "Tuesday-WorkingHours.pcap_ISCX.csv"

TEST_FILES = [
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


# ============================================================
# CONSTANT FEATURES
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
# HELPER
# ============================================================

def load_csv(filename):
    """Load one CIC-IDS2017 CSV and normalize column names."""

    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}"
        )

    df = pd.read_csv(path)

    # CIC-IDS2017 CSVs may contain leading/trailing
    # whitespace in column names.
    df.columns = df.columns.astype(str).str.strip()

    if "Label" not in df.columns:
        raise ValueError(
            f"'Label' column not found in {filename}. "
            f"Available columns include: {list(df.columns[-10:])}"
        )

    return df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LOADING CIC-IDS2017")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. MONDAY → TRAINING
    # --------------------------------------------------------

    monday_df = load_csv(MONDAY)

    monday_normal = monday_df[
        monday_df["Label"]
        .astype(str)
        .str.strip()
        .eq("BENIGN")
    ].copy()

    print(f"Monday total rows:    {len(monday_df):,}")
    print(f"Monday normal rows:   {len(monday_normal):,}")

    # --------------------------------------------------------
    # 2. TUESDAY → CALIBRATION
    # --------------------------------------------------------

    tuesday_df = load_csv(TUESDAY)

    tuesday_normal = tuesday_df[
        tuesday_df["Label"]
        .astype(str)
        .str.strip()
        .eq("BENIGN")
    ].copy()

    print(f"Tuesday total rows:   {len(tuesday_df):,}")
    print(f"Tuesday normal rows:  {len(tuesday_normal):,}")

    # --------------------------------------------------------
    # 3. WEDNESDAY-FRIDAY → UNTOUCHED TEST
    # --------------------------------------------------------

    test_dfs = []

    for filename in TEST_FILES:

        df = load_csv(filename)

        print(f"{filename}: {len(df):,} rows")

        test_dfs.append(df)

    test_df = pd.concat(
        test_dfs,
        ignore_index=True
    )

    test_labels = (
        test_df["Label"]
        .astype(str)
        .str.strip()
    )

    # BENIGN = 0
    # Everything else = 1
    test_y = (
        test_labels
        .ne("BENIGN")
        .astype(int)
        .to_numpy()
    )

    print(f"Test total rows:     {len(test_df):,}")
    print(
        f"Test normal rows:    {(test_y == 0).sum():,}"
    )
    print(
        f"Test attack rows:    {(test_y == 1).sum():,}"
    )

    # --------------------------------------------------------
    # 4. CREATE X MATRICES
    # --------------------------------------------------------

    train_X = monday_normal.drop(
        columns=["Label"]
    )

    calibration_X = tuesday_normal.drop(
        columns=["Label"]
    )

    test_X = test_df.drop(
        columns=["Label"]
    )

    # --------------------------------------------------------
    # 5. REMOVE CONSTANT FEATURES
    # --------------------------------------------------------

    train_X = train_X.drop(
        columns=CONSTANT_FEATURES,
        errors="ignore"
    )

    calibration_X = calibration_X.drop(
        columns=CONSTANT_FEATURES,
        errors="ignore"
    )

    test_X = test_X.drop(
        columns=CONSTANT_FEATURES,
        errors="ignore"
    )

    # --------------------------------------------------------
    # 6. LOCK FEATURE ORDER
    # --------------------------------------------------------

    feature_names = list(train_X.columns)

    calibration_X = calibration_X[
        feature_names
    ]

    test_X = test_X[
        feature_names
    ]

    print()
    print(
        f"Final feature count:  {len(feature_names)}"
    )

    # --------------------------------------------------------
    # 7. CONVERT TO NUMERIC
    # --------------------------------------------------------

    train_X = train_X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    calibration_X = calibration_X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    test_X = test_X.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # 8. REPLACE INFINITY WITH NaN
    # --------------------------------------------------------

    train_X = train_X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    calibration_X = calibration_X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    test_X = test_X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # --------------------------------------------------------
    # 9. MEDIAN IMPUTATION
    # --------------------------------------------------------
    #
    # FIT ONLY ON TRAINING DATA.
    # --------------------------------------------------------

    imputer = SimpleImputer(
        strategy="median"
    )

    train_X = imputer.fit_transform(
        train_X
    )

    calibration_X = imputer.transform(
        calibration_X
    )

    test_X = imputer.transform(
        test_X
    )

    # --------------------------------------------------------
    # 10. STANDARD SCALING
    # --------------------------------------------------------
    #
    # FIT ONLY ON TRAINING DATA.
    # --------------------------------------------------------

    scaler = StandardScaler()

    train_X = scaler.fit_transform(
        train_X
    )

    calibration_X = scaler.transform(
        calibration_X
    )

    test_X = scaler.transform(
        test_X
    )

    # --------------------------------------------------------
    # 11. SAVE PREPROCESSOR
    # --------------------------------------------------------

    preprocessor = {
        "imputer": imputer,
        "scaler": scaler,
        "feature_names": feature_names,
        "removed_constant_features": CONSTANT_FEATURES,
    }

    preprocessor_path = (
        MODEL_DIR /
        "cic2017_preprocessor.joblib"
    )

    joblib.dump(
        preprocessor,
        preprocessor_path
    )

    # --------------------------------------------------------
    # 12. SAVE PROCESSED DATA
    # --------------------------------------------------------

    np.save(
        MODEL_DIR / "X_train.npy",
        train_X
    )

    np.save(
        MODEL_DIR / "X_calibration.npy",
        calibration_X
    )

    np.save(
        MODEL_DIR / "X_test.npy",
        test_X
    )

    np.save(
        MODEL_DIR / "y_test.npy",
        test_y
    )

    # --------------------------------------------------------
    # 13. CREATE REPORT
    # --------------------------------------------------------

    report = {
        "dataset": "CIC-IDS2017",

        "strategy": "day_based_normal_training",

        "feature_count": len(feature_names),

        "removed_constant_features":
            CONSTANT_FEATURES,

        "training": {
            "source": [
                MONDAY
            ],
            "rows": int(len(train_X)),
            "normal_rows": int(len(train_X)),
            "attack_rows": 0,
        },

        "calibration": {
            "source": [
                TUESDAY
            ],
            "rows": int(len(calibration_X)),
            "normal_rows": int(len(calibration_X)),
            "attack_rows": 0,
        },

        "test": {
            "source": TEST_FILES,
            "rows": int(len(test_X)),
            "normal_rows": int(
                (test_y == 0).sum()
            ),
            "attack_rows": int(
                (test_y == 1).sum()
            ),
        },

        "preprocessing": [
            "Remove Label",
            "Remove constant features",
            "Replace infinity with NaN",
            "Median imputation fitted on training data only",
            "StandardScaler fitted on training data only",
        ],

        "split_policy": {
            "training":
                "Monday BENIGN only",

            "calibration":
                "Tuesday BENIGN only",

            "test":
                "Wednesday-Friday all traffic",

            "tuesday_attack_traffic":
                "Excluded from primary test set",
        },

        "feature_names": feature_names,
    }

    report_path = (
        REPORT_DIR /
        "cic2017_preprocessing.json"
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

    # --------------------------------------------------------
    # 14. FINAL OUTPUT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("CIC-IDS2017 PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Final features:       {len(feature_names)}"
    )

    print(
        f"Training normal:      {len(train_X):,}"
    )

    print(
        f"Calibration normal:   {len(calibration_X):,}"
    )

    print(
        f"Test rows:             {len(test_X):,}"
    )

    print(
        f"Test normal:           {(test_y == 0).sum():,}"
    )

    print(
        f"Test attacks:          {(test_y == 1).sum():,}"
    )

    print()
    print("Split:")
    print(f"  TRAIN       → {MONDAY}")
    print(f"  CALIBRATION → {TUESDAY}")
    print("  TEST        → Wednesday-Friday")

    print()
    print("Preprocessor saved:")
    print(preprocessor_path)

    print()
    print("Processed arrays saved:")
    print(
        MODEL_DIR / "X_train.npy"
    )
    print(
        MODEL_DIR / "X_calibration.npy"
    )
    print(
        MODEL_DIR / "X_test.npy"
    )
    print(
        MODEL_DIR / "y_test.npy"
    )

    print()
    print("Report saved:")
    print(report_path)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()