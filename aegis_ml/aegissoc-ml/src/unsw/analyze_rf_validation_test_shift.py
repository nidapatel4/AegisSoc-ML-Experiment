from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = ROOT / "data" / "UNSW_NB15_training-set.csv"
TEST_PATH = ROOT / "data" / "UNSW_NB15_testing-set.csv"

MODEL_PATH = ROOT / "models" / "unsw_random_forest.joblib"
PREPROCESSOR_PATH = ROOT / "models" / "unsw_random_forest_preprocessor.joblib"

SKEWED_COLUMNS = [
    "spkts", "dpkts", "sbytes", "dbytes", "sloss", "dloss",
    "dinpkt", "sjit", "djit", "trans_depth",
    "response_body_len", "ct_flw_http_mthd"
]

print("Loading model and preprocessor...")
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

print("Loading datasets...")
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

# Recreate the exact training/validation split
from sklearn.model_selection import train_test_split

X = train.drop(columns=["id", "attack_cat", "label"])
y = train["label"].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

# Apply same Log1p transformation
for col in SKEWED_COLUMNS:
    X_val[col] = np.log1p(X_val[col].clip(lower=0))

X_test = test.drop(columns=["id", "attack_cat", "label"])

for col in SKEWED_COLUMNS:
    X_test[col] = np.log1p(X_test[col].clip(lower=0))

print("\nTransforming validation and test data...")

X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

print(f"Validation shape: {X_val_processed.shape}")
print(f"Test shape:       {X_test_processed.shape}")

print("\nGenerating probabilities...")

val_prob = model.predict_proba(X_val_processed)[:, 1]
test_prob = model.predict_proba(X_test_processed)[:, 1]

# Only NORMAL samples matter for FPR calibration
val_normal_prob = val_prob[y_val.values == 0]
test_normal_prob = test_prob[test["label"].values == 0]

print("\n" + "=" * 70)
print("NORMAL-SAMPLE PROBABILITY DISTRIBUTION")
print("=" * 70)

def show_stats(name, values):
    print(f"\n{name}")
    print(f"Count:  {len(values)}")
    print(f"Mean:   {np.mean(values):.6f}")
    print(f"Std:    {np.std(values):.6f}")
    print(f"Min:    {np.min(values):.6f}")
    print(f"25%:    {np.percentile(values, 25):.6f}")
    print(f"50%:    {np.percentile(values, 50):.6f}")
    print(f"75%:    {np.percentile(values, 75):.6f}")
    print(f"90%:    {np.percentile(values, 90):.6f}")
    print(f"95%:    {np.percentile(values, 95):.6f}")
    print(f"99%:    {np.percentile(values, 99):.6f}")
    print(f"Max:    {np.max(values):.6f}")

show_stats("VALIDATION NORMALS", val_normal_prob)
show_stats("TEST NORMALS", test_normal_prob)


# ---------------------------------------------------------
# Check actual FPR at important thresholds
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("NORMAL FPR COMPARISON")
print("=" * 70)

thresholds = [0.50, 0.60, 0.63, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

for threshold in thresholds:

    val_fpr = np.mean(val_normal_prob >= threshold)
    test_fpr = np.mean(test_normal_prob >= threshold)

    print(
        f"Threshold {threshold:.2f} | "
        f"Validation FPR {val_fpr * 100:.2f}% | "
        f"Test FPR {test_fpr * 100:.2f}%"
    )


# ---------------------------------------------------------
# Compare categorical distributions
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("CATEGORICAL DISTRIBUTION CHECK")
print("=" * 70)

for col in ["proto", "service", "state"]:

    print(f"\n--- {col} ---")

    val_counts = X_val[col].value_counts(normalize=True)
    test_counts = X_test[col].value_counts(normalize=True)

    combined = pd.DataFrame({
        "validation": val_counts,
        "test": test_counts
    }).fillna(0)

    combined["absolute_difference"] = (
        combined["validation"] - combined["test"]
    ).abs()

    top_changes = combined.sort_values(
        "absolute_difference",
        ascending=False
    ).head(10)

    print(top_changes)


print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)