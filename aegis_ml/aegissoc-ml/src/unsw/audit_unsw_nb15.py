import json
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "data" / "UNSW_NB15_training-set.csv"
TEST_PATH = BASE_DIR / "data" / "UNSW_NB15_testing-set.csv"

REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

AUDIT_PATH = REPORTS_DIR / "unsw_nb15_data_audit.json"
DISTRIBUTION_PATH = REPORTS_DIR / "unsw_nb15_class_distribution.csv"


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)


# --------------------------------------------------
# Audit one dataset
# --------------------------------------------------

def audit_dataset(df, name):
    audit = {
        "dataset": name,

        "rows": int(df.shape[0]),

        "columns": int(df.shape[1]),

        "column_names": df.columns.tolist(),

        "data_types": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },

        "missing_values": {
            column: int(count)
            for column, count in df.isnull().sum().items()
            if count > 0
        },

        "total_missing_values": int(df.isnull().sum().sum()),

        "duplicate_rows": int(df.duplicated().sum()),

        "normal_count": int(
            (df["label"] == 0).sum()
        ),

        "attack_count": int(
            (df["label"] == 1).sum()
        ),

        "attack_categories": (
            df.loc[df["label"] == 1, "attack_cat"]
            .value_counts()
            .to_dict()
        ),

        "class_distribution": {
            "normal": float((df["label"] == 0).mean()),
            "attack": float((df["label"] == 1).mean())
        }
    }

    return audit


# --------------------------------------------------
# Run audits
# --------------------------------------------------

train_audit = audit_dataset(
    train_df,
    "UNSW_NB15_training-set"
)

test_audit = audit_dataset(
    test_df,
    "UNSW_NB15_testing-set"
)


# --------------------------------------------------
# Combined class distribution
# --------------------------------------------------

distribution_rows = []

for dataset_name, df in [
    ("training", train_df),
    ("testing", test_df)
]:

    normal_count = int((df["label"] == 0).sum())
    attack_count = int((df["label"] == 1).sum())

    distribution_rows.append({
        "dataset": dataset_name,
        "class": "normal",
        "count": normal_count,
        "percentage": normal_count / len(df) * 100
    })

    distribution_rows.append({
        "dataset": dataset_name,
        "class": "attack",
        "count": attack_count,
        "percentage": attack_count / len(df) * 100
    })

    for category, count in (
        df.loc[df["label"] == 1, "attack_cat"]
        .value_counts()
        .items()
    ):
        distribution_rows.append({
            "dataset": dataset_name,
            "class": "attack_category",
            "attack_category": str(category),
            "count": int(count),
            "percentage": count / len(df) * 100
        })


distribution_df = pd.DataFrame(distribution_rows)


# --------------------------------------------------
# Save JSON report
# --------------------------------------------------

audit_report = {
    "dataset": "UNSW-NB15",

    "source": "Official UNSW-NB15 training/testing CSV partitions",

    "training": train_audit,

    "testing": test_audit
}

with open(AUDIT_PATH, "w", encoding="utf-8") as f:
    json.dump(
        audit_report,
        f,
        indent=2,
        default=str
    )


# --------------------------------------------------
# Save class distribution CSV
# --------------------------------------------------

distribution_df.to_csv(
    DISTRIBUTION_PATH,
    index=False
)


# --------------------------------------------------
# Console summary
# --------------------------------------------------

print("\nUNSW-NB15 DATASET AUDIT")
print("=" * 50)

for audit in [train_audit, test_audit]:

    print(f"\n{audit['dataset']}")
    print("-" * 50)

    print("Rows:", audit["rows"])
    print("Columns:", audit["columns"])
    print("Missing values:", audit["total_missing_values"])
    print("Duplicate rows:", audit["duplicate_rows"])
    print("Normal:", audit["normal_count"])
    print("Attack:", audit["attack_count"])

    print("\nAttack categories:")
    for category, count in audit["attack_categories"].items():
        print(f"  {category}: {count}")

print("\nReports created:")
print(AUDIT_PATH)
print(DISTRIBUTION_PATH)