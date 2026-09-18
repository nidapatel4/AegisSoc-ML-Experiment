import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

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

FEATURES = COLUMNS[:-2]

DOS = {
    "back", "land", "neptune", "pod", "smurf",
    "teardrop", "mailbomb", "processtable",
    "udpstorm", "apache2", "worm"
}

PROBE = {
    "ipsweep", "nmap", "portsweep", "satan",
    "mscan", "saint"
}

R2L = {
    "ftp_write", "guess_passwd", "imap", "multihop",
    "phf", "spy", "warezclient", "warezmaster",
    "named", "sendmail", "snmpgetattack", "snmpguess",
    "xlock", "xsnoop", "httptunnel"
}

U2R = {
    "buffer_overflow", "loadmodule", "perl",
    "rootkit", "ps", "sqlattack", "xterm"
}


# ============================================================
# LOAD
# ============================================================

print("Loading data...")

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

normal = train[
    train["label"] == "normal"
].copy()

attacks = train[
    train["label"] != "normal"
].copy()

fit_normal, calibration_normal = train_test_split(
    normal,
    test_size=0.25,
    random_state=RANDOM_STATE
)

# Separate validation attacks from the final test.
validation_attack = attacks.copy()


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = build_preprocessor()

X_fit = preprocessor.fit_transform(
    fit_normal
)

X_calib = preprocessor.transform(
    calibration_normal
)

X_validation_attack = preprocessor.transform(
    validation_attack
)

X_test = preprocessor.transform(
    test
)


# ============================================================
# TUNED ISOLATION FOREST
# ============================================================

print("\nTraining tuned Isolation Forest...")

model = IsolationForest(
    n_estimators=800,
    max_samples=1.0,
    max_features=0.7,
    contamination=0.01,
    bootstrap=False,
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_fit)

calibration_scores = model.decision_function(
    X_calib
)

validation_attack_scores = model.decision_function(
    X_validation_attack
)

test_scores = model.decision_function(
    X_test
)


# ============================================================
# VALIDATION DATA
# ============================================================

validation_scores = np.concatenate([
    calibration_scores,
    validation_attack_scores
])

validation_labels = np.concatenate([
    np.zeros(len(calibration_scores)),
    np.ones(len(validation_attack_scores))
])


# ============================================================
# THRESHOLD SWEEP
# ============================================================

target_fprs = np.arange(
    0.005,
    0.201,
    0.005
)

results = []

for target_fpr in target_fprs:

    threshold = np.quantile(
        calibration_scores,
        target_fpr
    )

    predictions = (
        validation_scores < threshold
    ).astype(int)

    precision = precision_score(
        validation_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        validation_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        validation_labels,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        validation_labels,
        predictions
    ).ravel()

    actual_fpr = fp / (fp + tn)

    attack_predictions = (
        validation_attack_scores < threshold
    )

    family = {}

    for name, labels in [
        ("DoS", DOS),
        ("Probe", PROBE),
        ("R2L", R2L),
        ("U2R", U2R)
    ]:

        mask = validation_attack[
            "label"
        ].isin(labels).to_numpy()

        family[name] = (
            attack_predictions[mask].sum()
            / mask.sum()
        )

    results.append({
        "target_fpr": target_fpr,
        "threshold": threshold,
        "actual_fpr": actual_fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "DoS": family["DoS"],
        "Probe": family["Probe"],
        "R2L": family["R2L"],
        "U2R": family["U2R"]
    })


results_df = pd.DataFrame(results)


# ============================================================
# VALIDATION RESULTS
# ============================================================

print("\n" + "=" * 110)
print("VALIDATION THRESHOLD SWEEP")
print("=" * 110)

print(
    results_df[
        [
            "target_fpr",
            "threshold",
            "actual_fpr",
            "precision",
            "recall",
            "f1",
            "DoS",
            "Probe",
            "R2L",
            "U2R"
        ]
    ].to_string(index=False)
)


# ============================================================
# FIND BEST F1
# ============================================================

best = results_df.loc[
    results_df["f1"].idxmax()
]

print("\n" + "=" * 110)
print("BEST VALIDATION F1")
print("=" * 110)

print(
    best.to_string()
)


# ============================================================
# FIND BEST F1 UNDER FPR CONSTRAINTS
# ============================================================

print("\n" + "=" * 110)
print("BEST F1 UNDER FPR CONSTRAINTS")
print("=" * 110)

for limit in [
    0.07,
    0.08,
    0.09,
    0.10,
    0.12,
    0.15,
    0.20
]:

    valid = results_df[
        results_df["actual_fpr"] <= limit
    ]

    if len(valid) == 0:
        continue

    row = valid.loc[
        valid["f1"].idxmax()
    ]

    print(
        f"FPR <= {limit:.0%} | "
        f"target={row.target_fpr:.1%} | "
        f"actual={row.actual_fpr:.2%} | "
        f"F1={row.f1:.4f} | "
        f"Recall={row.recall:.4f} | "
        f"Precision={row.precision:.4f} | "
        f"R2L={row.R2L:.4f} | "
        f"U2R={row.U2R:.4f}"
    )


# ============================================================
# SELECT THRESHOLDS USING VALIDATION ONLY
# ============================================================

selected = []

for limit in [
    0.07,
    0.08,
    0.09,
    0.10,
    0.12,
    0.15,
    0.20
]:

    valid = results_df[
        results_df["actual_fpr"] <= limit
    ]

    if len(valid):

        row = valid.loc[
            valid["f1"].idxmax()
        ]

        selected.append(row)


selected_df = pd.DataFrame(
    selected
).drop_duplicates(
    subset=["threshold"]
)


# ============================================================
# FINAL TEST
#
# ONLY NOW touch KDDTest+
# ============================================================

print("\n" + "=" * 110)
print("UNTOUCHED KDDTest+")
print("=" * 110)

test_labels = (
    test["label"] != "normal"
).astype(int).to_numpy()

test_results = []

for _, row in selected_df.iterrows():

    threshold = row["threshold"]

    predictions = (
        test_scores < threshold
    ).astype(int)

    precision = precision_score(
        test_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        test_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        test_labels,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        test_labels,
        predictions
    ).ravel()

    actual_fpr = fp / (fp + tn)

    family = {}

    for name, labels in [
        ("DoS", DOS),
        ("Probe", PROBE),
        ("R2L", R2L),
        ("U2R", U2R)
    ]:

        mask = test[
            "label"
        ].isin(labels).to_numpy()

        family[name] = (
            predictions[mask].sum()
            / mask.sum()
        )

    unseen = {
        "apache2", "httptunnel", "mailbomb",
        "mscan", "named", "processtable",
        "ps", "sendmail", "snmpgetattack",
        "snmpguess", "sqlattack", "udpstorm",
        "worm", "xlock", "xsnoop"
    }

    unseen_mask = test[
        "label"
    ].isin(unseen).to_numpy()

    unseen_recall = (
        predictions[unseen_mask].sum()
        / unseen_mask.sum()
    )

    test_results.append({
        "validation_target_fpr": row["target_fpr"],
        "threshold": threshold,
        "actual_fpr": actual_fpr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "DoS": family["DoS"],
        "Probe": family["Probe"],
        "R2L": family["R2L"],
        "U2R": family["U2R"],
        "unseen": unseen_recall
    })


test_df_results = pd.DataFrame(
    test_results
)

print(
    test_df_results.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

Path("reports").mkdir(
    exist_ok=True
)

results_df.to_json(
    "reports/threshold_validation.json",
    orient="records",
    indent=2
)

test_df_results.to_json(
    "reports/threshold_final_test.json",
    orient="records",
    indent=2
)

print("\nSaved:")
print("reports/threshold_validation.json")
print("reports/threshold_final_test.json")