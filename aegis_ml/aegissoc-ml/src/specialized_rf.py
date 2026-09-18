import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import train_test_split

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


# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------

train_df = pd.read_csv(
    DATA / "KDDTrain+.txt",
    header=None,
    names=COLUMNS
)

test_df = pd.read_csv(
    DATA / "KDDTest+.txt",
    header=None,
    names=COLUMNS
)


# ---------------------------------------------------------
# R2L / U2R labels
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Build specialized training set
#
# 0 = Normal
# 1 = R2L/U2R
# ---------------------------------------------------------

normal_df = train_df[
    train_df["label"] == "normal"
].copy()

special_df = train_df[
    train_df["label"].isin(r2l_labels | u2r_labels)
].copy()

special_df["target"] = 1
normal_df["target"] = 0

# Balance the training data so normal traffic does not
# completely dominate the small R2L/U2R population.

normal_sample = normal_df.sample(
    n=min(len(normal_df), len(special_df) * 3),
    random_state=RANDOM_STATE
)

training_df = pd.concat(
    [
        normal_sample,
        special_df
    ],
    ignore_index=True
)

print("=" * 90)
print("SPECIALIZED R2L/U2R RANDOM FOREST")
print("=" * 90)

print(f"Normal samples used : {len(normal_sample)}")
print(f"R2L/U2R samples     : {len(special_df)}")
print(f"Total training      : {len(training_df)}")


# ---------------------------------------------------------
# Split training data
# ---------------------------------------------------------

train_part, validation_part = train_test_split(
    training_df,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=training_df["target"]
)


# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------

preprocessor = build_preprocessor()

X_train = preprocessor.fit_transform(
    train_part
)

X_validation = preprocessor.transform(
    validation_part
)

X_test = preprocessor.transform(
    test_df
)

y_train = train_part["target"].to_numpy()
y_validation = validation_part["target"].to_numpy()


# ---------------------------------------------------------
# Random Forest
# ---------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training complete.")


# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------

validation_pred = model.predict(
    X_validation
)

print("\n" + "=" * 90)
print("VALIDATION RESULTS")
print("=" * 90)

print(
    f"Precision : "
    f"{precision_score(y_validation, validation_pred):.4f}"
)

print(
    f"Recall    : "
    f"{recall_score(y_validation, validation_pred):.4f}"
)

print(
    f"F1        : "
    f"{f1_score(y_validation, validation_pred):.4f}"
)


# ---------------------------------------------------------
# Evaluate specifically on KDDTest+
# ---------------------------------------------------------

test_labels = test_df["label"]

actual_special = test_labels.isin(
    r2l_labels | u2r_labels
).astype(int).to_numpy()

test_pred = model.predict(X_test)

print("\n" + "=" * 90)
print("KDDTest+ R2L/U2R RESULTS")
print("=" * 90)

precision = precision_score(
    actual_special,
    test_pred,
    zero_division=0
)

recall = recall_score(
    actual_special,
    test_pred,
    zero_division=0
)

f1 = f1_score(
    actual_special,
    test_pred,
    zero_division=0
)

tn, fp, fn, tp = confusion_matrix(
    actual_special,
    test_pred,
    labels=[0, 1]
).ravel()

fpr = fp / (fp + tn)

print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1        : {f1:.4f}")
print(f"FPR       : {fpr:.4f}")

print("\nConfusion Matrix:")
print(
    confusion_matrix(
        actual_special,
        test_pred
    )
)


# ---------------------------------------------------------
# R2L vs U2R separately
# ---------------------------------------------------------

for name, labels in [
    ("R2L", r2l_labels),
    ("U2R", u2r_labels)
]:

    mask = test_labels.isin(labels)

    if mask.sum() == 0:
        continue

    recall_family = (
        test_pred[mask.to_numpy()].sum()
        / mask.sum()
    )

    print(
        f"{name} Recall : "
        f"{recall_family:.4f} "
        f"({test_pred[mask.to_numpy()].sum()}/{mask.sum()})"
    )


# ---------------------------------------------------------
# Feature importance
# ---------------------------------------------------------

feature_names = preprocessor.get_feature_names_out()

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\n" + "=" * 90)
print("TOP 20 FEATURES")
print("=" * 90)

print(
    importance.head(20).to_string(
        index=False
    )
)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

Path("reports").mkdir(exist_ok=True)

importance.head(50).to_json(
    "reports/specialized_rf_feature_importance.json",
    orient="records",
    indent=2
)

print("\nSaved:")
print("reports/specialized_rf_feature_importance.json")