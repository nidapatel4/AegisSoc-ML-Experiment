import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from preprocessing import build_preprocessor


RANDOM_STATE = 42

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


# =========================================================
# LOAD
# =========================================================

train = pd.read_csv(
    DATA / "KDDTrain+.txt",
    header=None,
    names=COLUMNS
)

test = pd.read_csv(
    DATA / "KDDTest+.txt",
    header=None,
    names=COLUMNS
)


# =========================================================
# ATTACK GROUPS
# =========================================================

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


# =========================================================
# SELECT R2L/U2R
# =========================================================

target_labels = r2l_labels | u2r_labels

train_special = train[
    train["label"].isin(target_labels)
].copy()

test_special = test[
    test["label"].isin(target_labels)
].copy()


print("=" * 100)
print("R2L / U2R DISTRIBUTION SHIFT ANALYSIS")
print("=" * 100)

print(f"\nTrain R2L/U2R: {len(train_special)}")
print(f"Test  R2L/U2R: {len(test_special)}")


# =========================================================
# ATTACK TYPE DISTRIBUTION
# =========================================================

print("\n" + "=" * 100)
print("ATTACK TYPE COUNTS")
print("=" * 100)

train_counts = train_special["label"].value_counts()
test_counts = test_special["label"].value_counts()

distribution = pd.DataFrame({
    "train_count": train_counts,
    "test_count": test_counts
}).fillna(0)

distribution["train_%"] = (
    distribution["train_count"]
    / distribution["train_count"].sum()
    * 100
)

distribution["test_%"] = (
    distribution["test_count"]
    / distribution["test_count"].sum()
    * 100
)

distribution["change_pp"] = (
    distribution["test_%"]
    - distribution["train_%"]
)

print(
    distribution
    .sort_values("test_count", ascending=False)
    .to_string()
)


# =========================================================
# NUMERIC FEATURE SHIFT
# =========================================================

numeric_features = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "hot",
    "num_failed_logins",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]


print("\n" + "=" * 100)
print("NUMERIC FEATURE DISTRIBUTION SHIFT")
print("=" * 100)

rows = []

for feature in numeric_features:

    train_values = train_special[feature].astype(float)
    test_values = test_special[feature].astype(float)

    train_mean = train_values.mean()
    test_mean = test_values.mean()

    train_median = train_values.median()
    test_median = test_values.median()

    train_std = train_values.std()
    test_std = test_values.std()

    pooled_std = np.sqrt(
        (train_std ** 2 + test_std ** 2) / 2
    )

    if pooled_std > 0:
        effect = abs(
            train_mean - test_mean
        ) / pooled_std
    else:
        effect = 0

    rows.append({
        "feature": feature,
        "train_mean": train_mean,
        "test_mean": test_mean,
        "train_median": train_median,
        "test_median": test_median,
        "effect_size": effect
    })


shift_df = pd.DataFrame(rows)

shift_df = shift_df.sort_values(
    "effect_size",
    ascending=False
)

print(
    shift_df.head(20).to_string(
        index=False
    )
)


# =========================================================
# R2L ONLY
# =========================================================

print("\n" + "=" * 100)
print("R2L NUMERIC SHIFT — TOP FEATURES")
print("=" * 100)

train_r2l = train[
    train["label"].isin(r2l_labels)
]

test_r2l = test[
    test["label"].isin(r2l_labels)
]

rows = []

for feature in numeric_features:

    a = train_r2l[feature].astype(float)
    b = test_r2l[feature].astype(float)

    pooled = np.sqrt(
        (a.std() ** 2 + b.std() ** 2) / 2
    )

    effect = (
        abs(a.mean() - b.mean()) / pooled
        if pooled > 0 else 0
    )

    rows.append({
        "feature": feature,
        "train_mean": a.mean(),
        "test_mean": b.mean(),
        "effect_size": effect
    })

print(
    pd.DataFrame(rows)
    .sort_values("effect_size", ascending=False)
    .head(15)
    .to_string(index=False)
)


# =========================================================
# U2R ONLY
# =========================================================

print("\n" + "=" * 100)
print("U2R NUMERIC SHIFT — TOP FEATURES")
print("=" * 100)

train_u2r = train[
    train["label"].isin(u2r_labels)
]

test_u2r = test[
    test["label"].isin(u2r_labels)
]

rows = []

for feature in numeric_features:

    a = train_u2r[feature].astype(float)
    b = test_u2r[feature].astype(float)

    pooled = np.sqrt(
        (a.std() ** 2 + b.std() ** 2) / 2
    )

    effect = (
        abs(a.mean() - b.mean()) / pooled
        if pooled > 0 else 0
    )

    rows.append({
        "feature": feature,
        "train_mean": a.mean(),
        "test_mean": b.mean(),
        "effect_size": effect
    })

print(
    pd.DataFrame(rows)
    .sort_values("effect_size", ascending=False)
    .head(15)
    .to_string(index=False)
)


# =========================================================
# CATEGORICAL SHIFT
# =========================================================

categorical_features = [
    "protocol_type",
    "service",
    "flag"
]

print("\n" + "=" * 100)
print("CATEGORICAL DISTRIBUTION SHIFT")
print("=" * 100)

for feature in categorical_features:

    print(f"\n--- {feature} ---")

    train_dist = (
        train_special[feature]
        .value_counts(normalize=True)
        .mul(100)
        .rename("train_%")
    )

    test_dist = (
        test_special[feature]
        .value_counts(normalize=True)
        .mul(100)
        .rename("test_%")
    )

    comparison = pd.concat(
        [train_dist, test_dist],
        axis=1
    ).fillna(0)

    comparison["change_pp"] = (
        comparison["test_%"]
        - comparison["train_%"]
    )

    print(
        comparison
        .sort_values(
            "test_%",
            ascending=False
        )
        .head(15)
        .to_string()
    )


# =========================================================
# IMPORTANT FEATURES FROM RF
# =========================================================

important = [
    "dst_host_srv_count",
    "src_bytes",
    "dst_host_same_src_port_rate",
    "count",
    "srv_count",
    "dst_host_count",
    "dst_bytes",
    "hot",
    "is_guest_login"
]

print("\n" + "=" * 100)
print("RF IMPORTANT FEATURES — TRAIN VS TEST")
print("=" * 100)

rows = []

for feature in important:

    a = train_special[feature].astype(float)
    b = test_special[feature].astype(float)

    rows.append({
        "feature": feature,
        "train_mean": a.mean(),
        "test_mean": b.mean(),
        "train_median": a.median(),
        "test_median": b.median()
    })

print(
    pd.DataFrame(rows)
    .to_string(index=False)
)


# =========================================================
# SAVE
# =========================================================

Path("reports").mkdir(exist_ok=True)

shift_df.to_json(
    "reports/r2l_u2r_distribution_shift.json",
    orient="records",
    indent=2
)

print("\nSaved:")
print("reports/r2l_u2r_distribution_shift.json")