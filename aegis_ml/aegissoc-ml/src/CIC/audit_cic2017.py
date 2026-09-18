import json
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "MachineLearningCVE"

REPORT_DIR = BASE_DIR / "reports" / "CIC"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Find CIC-IDS2017 CSV files
# ---------------------------------------------------------

csv_files = sorted(DATA_DIR.glob("*.csv"))

if not csv_files:
    raise FileNotFoundError(
        f"No CSV files found in: {DATA_DIR}"
    )


# ---------------------------------------------------------
# Audit
# ---------------------------------------------------------

report = {
    "dataset": "CIC-IDS2017",
    "data_directory": str(DATA_DIR),
    "files": [],
    "combined": {}
}

all_frames = []

for csv_file in csv_files:

    print("\n" + "=" * 70)
    print(f"FILE: {csv_file.name}")
    print("=" * 70)

    df = pd.read_csv(csv_file, low_memory=False)

    all_frames.append(df)

    # Remove accidental whitespace from column names
    df.columns = df.columns.str.strip()

    # Label column
    label_col = "Label"

    if label_col in df.columns:
        label_counts = (
            df[label_col]
            .astype(str)
            .str.strip()
            .value_counts()
            .to_dict()
        )
    else:
        label_counts = {}

    file_report = {
        "file": csv_file.name,
        "rows": int(len(df)),
        "columns_count": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {
            column: str(dtype)
            for column, dtype in df.dtypes.items()
        },
        "missing_values": {
            column: int(count)
            for column, count in df.isna().sum().items()
            if count > 0
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "label_counts": label_counts
    }

    report["files"].append(file_report)

    print(f"Rows:              {len(df):,}")
    print(f"Columns:           {len(df.columns)}")
    print(f"Duplicate rows:    {df.duplicated().sum():,}")

    print("\nLabel distribution:")
    if label_counts:
        for label, count in label_counts.items():
            print(f"  {label}: {count:,}")
    else:
        print("  Label column not found")

    missing_total = int(df.isna().sum().sum())
    print(f"\nTotal missing values: {missing_total:,}")

    if missing_total:
        print("Missing by column:")
        for column, count in df.isna().sum().items():
            if count > 0:
                print(f"  {column}: {count:,}")


# ---------------------------------------------------------
# Combined dataset audit
# ---------------------------------------------------------

print("\n\n" + "#" * 70)
print("COMBINED CIC-IDS2017 AUDIT")
print("#" * 70)

combined = pd.concat(all_frames, ignore_index=True)

label_col = "Label"

if label_col in combined.columns:

    combined[label_col] = (
        combined[label_col]
        .astype(str)
        .str.strip()
    )

    label_counts = combined[label_col].value_counts()

    benign_count = int(
        label_counts.get("BENIGN", 0)
    )

    attack_count = int(
        len(combined) - benign_count
    )

    report["combined"] = {
        "rows": int(len(combined)),
        "columns_count": int(len(combined.columns)),
        "columns": list(combined.columns),
        "duplicate_rows": int(combined.duplicated().sum()),
        "total_missing_values": int(
            combined.isna().sum().sum()
        ),
        "benign_rows": benign_count,
        "attack_rows": attack_count,
        "label_counts": {
            str(label): int(count)
            for label, count in label_counts.items()
        }
    }

else:

    report["combined"] = {
        "rows": int(len(combined)),
        "columns_count": int(len(combined.columns)),
        "columns": list(combined.columns),
        "duplicate_rows": int(combined.duplicated().sum()),
        "total_missing_values": int(
            combined.isna().sum().sum()
        ),
        "error": "Label column not found"
    }


# ---------------------------------------------------------
# Print combined summary
# ---------------------------------------------------------

print(f"\nTotal rows:          {len(combined):,}")
print(f"Total columns:       {len(combined.columns)}")
print(f"Duplicate rows:      {combined.duplicated().sum():,}")
print(
    f"Missing values:      "
    f"{combined.isna().sum().sum():,}"
)

if label_col in combined.columns:

    print("\nAll labels:")

    for label, count in combined[label_col].value_counts().items():
        print(f"  {label}: {count:,}")

    benign = (
        combined[label_col] == "BENIGN"
    ).sum()

    attacks = len(combined) - benign

    print(f"\nBENIGN:              {benign:,}")
    print(f"ATTACK:              {attacks:,}")


# ---------------------------------------------------------
# Save report
# ---------------------------------------------------------

output_file = (
    REPORT_DIR / "cic2017_data_audit.json"
)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(
        report,
        f,
        indent=2
    )


print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
print(f"Report saved to:")
print(output_file)