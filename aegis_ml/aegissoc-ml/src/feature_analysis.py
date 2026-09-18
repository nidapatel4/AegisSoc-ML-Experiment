import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from preprocessing import build_preprocessor


DATA = Path("data")

COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty"
]

train_df = pd.read_csv(
    DATA / "KDDTrain+.txt",
    header=None,
    names=COLUMNS
)

# ---------------------------------------------------------
# Create groups
# ---------------------------------------------------------

normal = train_df[train_df["label"] == "normal"]

r2l_labels = {
    "ftp_write", "guess_passwd", "imap", "multihop",
    "phf", "spy", "warezclient", "warezmaster",
    "named", "sendmail", "snmpgetattack", "snmpguess",
    "xlock", "xsnoop", "httptunnel"
}

u2r_labels = {
    "buffer_overflow", "loadmodule", "perl",
    "rootkit", "ps", "sqlattack", "xterm"
}

r2l = train_df[train_df["label"].isin(r2l_labels)]
u2r = train_df[train_df["label"].isin(u2r_labels)]

print("=" * 100)
print("R2L / U2R FEATURE SEPARATION ANALYSIS")
print("=" * 100)

print(f"Normal: {len(normal)}")
print(f"R2L   : {len(r2l)}")
print(f"U2R   : {len(u2r)}")


# ---------------------------------------------------------
# Numeric features
# ---------------------------------------------------------

numeric_features = [
    c for c in COLUMNS
    if c not in ["protocol_type", "service", "flag", "label", "difficulty"]
]


def analyze_group(name, attack_df):

    rows = []

    for feature in numeric_features:

        normal_values = pd.to_numeric(
            normal[feature],
            errors="coerce"
        ).fillna(0)

        attack_values = pd.to_numeric(
            attack_df[feature],
            errors="coerce"
        ).fillna(0)

        normal_mean = normal_values.mean()
        attack_mean = attack_values.mean()

        normal_median = normal_values.median()
        attack_median = attack_values.median()

        normal_std = normal_values.std()

        # Standardized mean difference
        if normal_std > 0:
            effect = abs(
                attack_mean - normal_mean
            ) / normal_std
        else:
            effect = 0

        # Difference in medians
        median_diff = abs(
            attack_median - normal_median
        )

        rows.append({
            "feature": feature,
            "normal_mean": normal_mean,
            "attack_mean": attack_mean,
            "normal_median": normal_median,
            "attack_median": attack_median,
            "effect_size": effect,
            "median_difference": median_diff,
        })

    result = pd.DataFrame(rows)

    result = result.sort_values(
        "effect_size",
        ascending=False
    )

    print("\n" + "=" * 100)
    print(f"TOP FEATURES SEPARATING NORMAL vs {name}")
    print("=" * 100)

    print(
        result.head(15).to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return result


r2l_results = analyze_group("R2L", r2l)
u2r_results = analyze_group("U2R", u2r)


# ---------------------------------------------------------
# Categorical feature distributions
# ---------------------------------------------------------

for feature in ["protocol_type", "service", "flag"]:

    print("\n" + "=" * 100)
    print(f"CATEGORICAL FEATURE: {feature}")
    print("=" * 100)

    normal_dist = (
        normal[feature]
        .value_counts(normalize=True)
        .rename("normal_pct")
    )

    r2l_dist = (
        r2l[feature]
        .value_counts(normalize=True)
        .rename("r2l_pct")
    )

    u2r_dist = (
        u2r[feature]
        .value_counts(normalize=True)
        .rename("u2r_pct")
    )

    combined = pd.concat(
        [normal_dist, r2l_dist, u2r_dist],
        axis=1
    ).fillna(0)

    combined["r2l_difference"] = abs(
        combined["r2l_pct"] -
        combined["normal_pct"]
    )

    combined["u2r_difference"] = abs(
        combined["u2r_pct"] -
        combined["normal_pct"]
    )

    print(
        combined.sort_values(
            "r2l_difference",
            ascending=False
        ).head(15).to_string()
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

Path("reports").mkdir(exist_ok=True)

r2l_results.to_json(
    "reports/r2l_feature_analysis.json",
    orient="records",
    indent=2
)

u2r_results.to_json(
    "reports/u2r_feature_analysis.json",
    orient="records",
    indent=2
)

print("\nSaved:")
print("reports/r2l_feature_analysis.json")
print("reports/u2r_feature_analysis.json")

