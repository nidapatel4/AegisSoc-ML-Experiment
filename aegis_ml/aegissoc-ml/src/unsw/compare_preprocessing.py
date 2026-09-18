from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "data" / "UNSW_NB15_training-set.csv"


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(TRAIN_PATH)

DROP_COLUMNS = ["id", "attack_cat", "label"]

X = df.drop(columns=DROP_COLUMNS)


CATEGORICAL_COLUMNS = [
    "proto",
    "service",
    "state"
]

NUMERICAL_COLUMNS = [
    column
    for column in X.columns
    if column not in CATEGORICAL_COLUMNS
]


# --------------------------------------------------
# Features suitable for log1p
# --------------------------------------------------

SKEWED_COLUMNS = [
    "spkts",
    "dpkts",
    "sbytes",
    "dbytes",
    "sloss",
    "dloss",
    "dinpkt",
    "sjit",
    "djit",
    "trans_depth",
    "response_body_len",
    "ct_flw_http_mthd"
]


# --------------------------------------------------
# Strategy A
# StandardScaler only
# --------------------------------------------------

standard_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            NUMERICAL_COLUMNS
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            CATEGORICAL_COLUMNS
        )
    ]
)


X_standard = standard_preprocessor.fit_transform(X)


# --------------------------------------------------
# Strategy B
# log1p + StandardScaler
# --------------------------------------------------

X_log = X.copy()

for column in SKEWED_COLUMNS:
    X_log[column] = np.log1p(X_log[column])


log_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            NUMERICAL_COLUMNS
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            CATEGORICAL_COLUMNS
        )
    ]
)


X_log_scaled = log_preprocessor.fit_transform(X_log)


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\nUNSW-NB15 PREPROCESSING COMPARISON")
print("=" * 70)

print("\nRaw feature count:")
print(f"  {len(X.columns)}")

print("\nNumerical features:")
print(f"  {len(NUMERICAL_COLUMNS)}")

print("\nCategorical features:")
print(f"  {len(CATEGORICAL_COLUMNS)}")

print("\nCategorical cardinality:")
for column in CATEGORICAL_COLUMNS:
    print(
        f"  {column}: "
        f"{X[column].nunique()} categories"
    )

print("\n" + "-" * 70)

print("\nStrategy A: StandardScaler only")
print(f"  Output dimensions: {X_standard.shape}")

print("\nStrategy B: log1p + StandardScaler")
print(f"  Output dimensions: {X_log_scaled.shape}")

print("\n" + "-" * 70)

print("\nLog-transformed features:")
for column in SKEWED_COLUMNS:
    print(f"  - {column}")

print("\nPreprocessing comparison complete.")