from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[2]

TRAIN_PATH = BASE_DIR / "data" / "UNSW_NB15_training-set.csv"

SKEWED_COLUMNS = [
    "spkts", "dpkts", "sbytes", "dbytes",
    "sloss", "dloss", "dinpkt", "sjit",
    "djit", "trans_depth", "response_body_len",
    "ct_flw_http_mthd",
]

DROP_COLUMNS = ["id", "attack_cat", "label"]

print("Loading UNSW-NB15...")
df = pd.read_csv(TRAIN_PATH)

# Separate normal traffic and attack families
normal = df[df["label"] == 0]

attack_families = [
    "Reconnaissance",
    "Fuzzers",
    "Shellcode",
    "DoS",
    "Exploits",
]

print(f"Normal records: {len(normal):,}")

# Only numerical model features
numeric_features = [
    col
    for col in df.columns
    if col not in DROP_COLUMNS
    and df[col].dtype != "object"
]

print(f"Numerical features: {len(numeric_features)}")

# ------------------------------------------------------------
# Calculate standardized mean difference
# ------------------------------------------------------------

results = []

for attack in attack_families:

    attack_df = df[df["attack_cat"] == attack]

    print("\n" + "=" * 70)
    print(f"{attack.upper()}")
    print("=" * 70)

    comparisons = []

    for feature in numeric_features:

        normal_values = normal[feature].astype(float)
        attack_values = attack_df[feature].astype(float)

        normal_mean = normal_values.mean()
        attack_mean = attack_values.mean()

        normal_std = normal_values.std()

        # Standardized difference from normal mean
        if normal_std == 0:
            effect = 0
        else:
            effect = abs(
                attack_mean - normal_mean
            ) / normal_std

        comparisons.append(
            {
                "feature": feature,
                "normal_mean": normal_mean,
                "attack_mean": attack_mean,
                "effect": effect,
            }
        )

    comparisons = sorted(
        comparisons,
        key=lambda x: x["effect"],
        reverse=True,
    )

    print(
        f"{'Feature':<25}"
        f"{'Normal Mean':>15}"
        f"{'Attack Mean':>15}"
        f"{'Difference':>15}"
    )

    print("-" * 70)

    for row in comparisons[:15]:

        print(
            f"{row['feature']:<25}"
            f"{row['normal_mean']:>15.3f}"
            f"{row['attack_mean']:>15.3f}"
            f"{row['effect']:>15.3f}"
        )

        results.append(
            {
                "attack_family": attack,
                **row,
            }
        )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output = pd.DataFrame(results)

REPORT_PATH = (
    BASE_DIR
    / "reports"
    / "unsw_nb15_feature_distribution_analysis.csv"
)

output.to_csv(REPORT_PATH, index=False)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

print(f"Report created:")
print(REPORT_PATH)
