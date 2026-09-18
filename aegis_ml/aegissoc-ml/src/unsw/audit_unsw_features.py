from pathlib import Path
import json

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "data" / "UNSW_NB15_training-set.csv"
REPORTS_DIR = BASE_DIR / "reports"

REPORTS_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = REPORTS_DIR / "unsw_nb15_feature_audit.json"


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

df = pd.read_csv(TRAIN_PATH)


# --------------------------------------------------
# Columns
# --------------------------------------------------

DROP_COLUMNS = ["id", "attack_cat", "label"]

FEATURE_COLUMNS = [
    column for column in df.columns
    if column not in DROP_COLUMNS
]

CATEGORICAL_COLUMNS = [
    "proto",
    "service",
    "state"
]

NUMERICAL_COLUMNS = [
    column
    for column in FEATURE_COLUMNS
    if column not in CATEGORICAL_COLUMNS
]


# --------------------------------------------------
# Numerical feature audit
# --------------------------------------------------

numerical_audit = {}

for column in NUMERICAL_COLUMNS:

    series = df[column]

    numerical_audit[column] = {
        "dtype": str(series.dtype),
        "unique_values": int(series.nunique()),
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "zero_count": int((series == 0).sum()),
        "zero_percentage": float((series == 0).mean() * 100),
        "constant": bool(series.nunique() <= 1)
    }


# --------------------------------------------------
# Categorical feature audit
# --------------------------------------------------

categorical_audit = {}

for column in CATEGORICAL_COLUMNS:

    series = df[column]

    value_counts = series.value_counts()

    categorical_audit[column] = {
        "dtype": str(series.dtype),
        "unique_values": int(series.nunique()),
        "most_common_values": {
            str(key): int(value)
            for key, value in value_counts.head(15).items()
        }
    }


# --------------------------------------------------
# Constant features
# --------------------------------------------------

constant_features = [
    column
    for column in FEATURE_COLUMNS
    if df[column].nunique() <= 1
]


# --------------------------------------------------
# Highly skewed numerical features
# --------------------------------------------------

skewness = df[NUMERICAL_COLUMNS].skew()

highly_skewed_features = {
    column: float(value)
    for column, value in skewness.items()
    if abs(value) > 10
}


# --------------------------------------------------
# Build report
# --------------------------------------------------

report = {
    "dataset": "UNSW-NB15",
    "rows_audited": int(len(df)),
    "total_raw_columns": int(len(df.columns)),

    "excluded_columns": DROP_COLUMNS,

    "feature_count": int(len(FEATURE_COLUMNS)),

    "categorical_features": CATEGORICAL_COLUMNS,

    "categorical_feature_count": int(len(CATEGORICAL_COLUMNS)),

    "numerical_feature_count": int(len(NUMERICAL_COLUMNS)),

    "constant_features": constant_features,

    "highly_skewed_features": highly_skewed_features,

    "numerical_features": numerical_audit,

    "categorical_features_detail": categorical_audit
}


# --------------------------------------------------
# Save report
# --------------------------------------------------

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)


# --------------------------------------------------
# Console summary
# --------------------------------------------------

print("\nUNSW-NB15 FEATURE QUALITY AUDIT")
print("=" * 70)

print(f"Rows audited: {len(df)}")
print(f"Raw columns: {len(df.columns)}")
print(f"Model features: {len(FEATURE_COLUMNS)}")
print(f"Numerical features: {len(NUMERICAL_COLUMNS)}")
print(f"Categorical features: {len(CATEGORICAL_COLUMNS)}")

print("\nConstant features:")
if constant_features:
    for column in constant_features:
        print(f"  - {column}")
else:
    print("  None")

print("\nHighly skewed numerical features (|skew| > 10):")
if highly_skewed_features:
    for column, value in highly_skewed_features.items():
        print(f"  - {column}: {value:.2f}")
else:
    print("  None")

print("\nCategorical cardinality:")
for column in CATEGORICAL_COLUMNS:
    print(
        f"  {column}: "
        f"{df[column].nunique()} unique values"
    )

print("\nReport created:")
print(OUTPUT_PATH)