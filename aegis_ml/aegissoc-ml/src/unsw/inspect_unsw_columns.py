from pathlib import Path
import pandas as pd


# Project root
BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "data" / "UNSW_NB15_training-set.csv"

# Load only the first few rows.
# We don't need to load the entire dataset just to inspect columns.
df = pd.read_csv(TRAIN_PATH, nrows=5)


print("\nUNSW-NB15 COLUMN INSPECTION")
print("=" * 70)

print(f"Total columns: {len(df.columns)}")

print("\nColumns and data types:")
print("-" * 70)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column:<25} {str(df[column].dtype)}")


print("\n" + "=" * 70)
print("COLUMN CATEGORIES")
print("=" * 70)

print("\nPotential identifiers:")
for column in ["id"]:
    if column in df.columns:
        print(f"  - {column}")


print("\nCategorical columns:")
for column in ["proto", "service", "state"]:
    if column in df.columns:
        print(f"  - {column}")


print("\nTarget / label columns:")
for column in ["attack_cat", "label"]:
    if column in df.columns:
        print(f"  - {column}")


print("\nPotential model input columns:")
excluded = {"id", "attack_cat", "label"}

model_columns = [
    column
    for column in df.columns
    if column not in excluded
]

print(f"  Total model input columns before preprocessing: {len(model_columns)}")

for column in model_columns:
    print(f"  - {column}")


print("\n" + "=" * 70)
print("SAMPLE VALUES")
print("=" * 70)

for column in df.columns:
    print(f"\n{column}:")
    print(df[column].tolist())