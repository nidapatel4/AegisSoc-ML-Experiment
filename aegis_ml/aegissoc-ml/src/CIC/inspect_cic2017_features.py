import json
from pathlib import Path

import pandas as pd


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "MachineLearningCVE"
REPORT_DIR = BASE_DIR / "reports" / "CIC"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# FIND CSV FILES
# =========================================================

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in: {DATA_DIR}"
    )


# =========================================================
# READ DATA
# =========================================================

print("Reading CIC-IDS2017 files...")

frames = []

for file in csv_files:
    print(f"  Loading: {file.name}")

    df = pd.read_csv(
        file,
        low_memory=False
    )

    # Remove whitespace around column names
    df.columns = df.columns.str.strip()

    frames.append(df)


# Combine only for feature inspection
df = pd.concat(
    frames,
    ignore_index=True
)


# =========================================================
# BASIC INFORMATION
# =========================================================

print("\n" + "=" * 70)
print("CIC-IDS2017 FEATURE AUDIT")
print("=" * 70)

print(f"Rows:    {len(df):,}")
print(f"Columns: {len(df.columns)}")


# =========================================================
# COLUMN INFORMATION
# =========================================================

column_info = []

for i, column in enumerate(df.columns):

    series = df[column]

    info = {
        "index": i,
        "name": column,
        "dtype": str(series.dtype),
        "missing_values": int(series.isna().sum()),
        "unique_values": int(series.nunique(dropna=True)),
        "is_numeric": bool(
            pd.api.types.is_numeric_dtype(series)
        )
    }

    # Numeric statistics
    if info["is_numeric"]:

        info["min"] = (
            float(series.min())
            if not series.dropna().empty
            else None
        )

        info["max"] = (
            float(series.max())
            if not series.dropna().empty
            else None
        )

        info["infinite_values"] = int(
            series.isin([float("inf"), float("-inf")]).sum()
        )

    else:

        # Show most common categorical values
        value_counts = (
            series.astype(str)
            .value_counts()
            .head(10)
            .to_dict()
        )

        info["top_values"] = {
            str(k): int(v)
            for k, v in value_counts.items()
        }

    column_info.append(info)


# =========================================================
# IDENTIFY COLUMN TYPES
# =========================================================

numeric_columns = [
    item["name"]
    for item in column_info
    if item["is_numeric"]
]

categorical_columns = [
    item["name"]
    for item in column_info
    if not item["is_numeric"]
]


# =========================================================
# POTENTIAL IDENTIFIER / LEAKAGE CANDIDATES
# =========================================================

identifier_keywords = [
    "id",
    "timestamp",
    "time",
    "date",
    "flow id",
    "source ip",
    "destination ip",
    "src ip",
    "dst ip"
]

identifier_candidates = []

for column in df.columns:

    column_lower = column.lower()

    if any(
        keyword in column_lower
        for keyword in identifier_keywords
    ):
        identifier_candidates.append(column)


# =========================================================
# LABEL INFORMATION
# =========================================================

label_column = None

for column in df.columns:

    if column.strip().lower() == "label":
        label_column = column
        break


label_values = {}

if label_column:

    label_counts = (
        df[label_column]
        .astype(str)
        .str.strip()
        .value_counts()
    )

    label_values = {
        str(label): int(count)
        for label, count in label_counts.items()
    }


# =========================================================
# PRINT RESULTS
# =========================================================

print("\nNUMERIC COLUMNS")
print("-" * 70)

for column in numeric_columns:
    print(column)


print("\nCATEGORICAL COLUMNS")
print("-" * 70)

for column in categorical_columns:
    print(column)


print("\nPOTENTIAL IDENTIFIER / TIME / LEAKAGE CANDIDATES")
print("-" * 70)

for column in identifier_candidates:
    print(column)


print("\nLABEL COLUMN")
print("-" * 70)

print(label_column)


print("\nLABEL VALUES")
print("-" * 70)

for label, count in label_values.items():
    print(f"{label}: {count:,}")


# =========================================================
# BUILD REPORT
# =========================================================

report = {
    "dataset": "CIC-IDS2017",

    "rows": int(len(df)),

    "columns_count": int(len(df.columns)),

    "columns": column_info,

    "numeric_columns": numeric_columns,

    "categorical_columns": categorical_columns,

    "potential_identifier_or_leakage_candidates":
        identifier_candidates,

    "label_column": label_column,

    "label_values": label_values
}


# =========================================================
# SAVE REPORT
# =========================================================

output_file = (
    REPORT_DIR /
    "cic2017_feature_audit.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


print("\n" + "=" * 70)
print("FEATURE AUDIT COMPLETE")
print("=" * 70)

print(f"Report saved to:")
print(output_file)