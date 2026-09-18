import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

sys.path.append(str(Path(__file__).resolve().parent))


# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
TARGET_FPR = 0.05

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

CATEGORICAL = [
    "protocol_type",
    "service",
    "flag"
]

NUMERIC = [
    f for f in FEATURES
    if f not in CATEGORICAL
]


# ============================================================
# FEATURE SETS
# ============================================================

FEATURE_SETS = {

    # --------------------------------------------------------
    # 1. CONTROL
    # Same 41 raw features used by original model.
    # --------------------------------------------------------
    "baseline_41": FEATURES,

    # --------------------------------------------------------
    # 2. BEHAVIORAL
    # Focus on traffic volume, rates, errors and activity.
    # --------------------------------------------------------
    "behavioral": [
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
        "is_guest_login",
        "count",
        "srv_count",
        "serror_rate",
        "srv_serror_rate",
        "rerror_rate",
        "srv_rerror_rate",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "protocol_type",
        "service",
        "flag",
    ],

    # --------------------------------------------------------
    # 3. R2L/U2R FOCUSED
    # Based directly on our feature analysis.
    # --------------------------------------------------------
    "minority_focused": [
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
        "is_guest_login",
        "count",
        "srv_count",
        "same_srv_rate",
        "diff_srv_rate",
        "srv_diff_host_rate",
        "dst_host_count",
        "dst_host_srv_count",
        "dst_host_same_srv_rate",
        "dst_host_diff_srv_rate",
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
        "protocol_type",
        "service",
        "flag",
    ],

    # --------------------------------------------------------
    # 4. REMOVE HOST-DOMINANT FEATURES
    # Tests whether host statistics are causing poor
    # generalization to KDDTest+.
    # --------------------------------------------------------
    "without_host_stats": [
        f for f in FEATURES
        if not f.startswith("dst_host_")
    ],

    # --------------------------------------------------------
    # 5. COMPACT HYBRID
    # Strong general traffic features + minority signals.
    # --------------------------------------------------------
    "compact_hybrid": [
        "duration",
        "protocol_type",
        "service",
        "flag",
        "src_bytes",
        "dst_bytes",
        "hot",
        "num_failed_logins",
        "num_compromised",
        "root_shell",
        "su_attempted",
        "num_root",
        "num_shells",
        "num_access_files",
        "is_guest_login",
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
        "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate",
    ],
}


# ============================================================
# ATTACK LABELS
# ============================================================

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
# LOAD DATA
# ============================================================

print("Loading NSL-KDD...")

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

normal_df = train_df[
    train_df["label"] == "normal"
].copy()

attack_df = train_df[
    train_df["label"] != "normal"
].copy()

fit_normal, calib_normal = train_test_split(
    normal_df,
    test_size=0.25,
    random_state=RANDOM_STATE
)

# Validation attacks come ONLY from KDDTrain+.
validation_attack = attack_df.copy()

print(f"Train shape : {train_df.shape}")
print(f"Test shape  : {test_df.shape}")
print(f"Normal fit  : {len(fit_normal)}")
print(f"Calibration: {len(calib_normal)}")
print(f"Attack val : {len(validation_attack)}")


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(feature_list):

    categorical = [
        f for f in feature_list
        if f in CATEGORICAL
    ]

    numeric = [
        f for f in feature_list
        if f not in CATEGORICAL
    ]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                numeric
            ),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical
            )
        ]
    )


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(
    feature_name,
    feature_list,
    use_test=False
):

    print("\n" + "=" * 100)
    print(f"FEATURE SET: {feature_name}")
    print(f"Raw features: {len(feature_list)}")
    print("=" * 100)

    start = time.time()

    preprocessor = build_preprocessor(
        feature_list
    )

    X_fit = preprocessor.fit_transform(
        fit_normal[feature_list]
    )

    X_calib = preprocessor.transform(
        calib_normal[feature_list]
    )

    X_attack = preprocessor.transform(
        validation_attack[feature_list]
    )

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

    calib_scores = model.decision_function(
        X_calib
    )

    threshold = np.quantile(
        calib_scores,
        TARGET_FPR
    )

    attack_scores = model.decision_function(
        X_attack
    )

    y_attack = np.ones(
        len(validation_attack),
        dtype=int
    )

    y_normal = np.zeros(
        len(calib_normal),
        dtype=int
    )

    y_true = np.concatenate(
        [y_normal, y_attack]
    )

    scores = np.concatenate(
        [calib_scores, attack_scores]
    )

    predictions = (
        scores < threshold
    ).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc = roc_auc_score(
        y_true,
        -scores
    )

    pr = average_precision_score(
        y_true,
        -scores
    )

    cm = confusion_matrix(
        y_true,
        predictions
    )

    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn)

    # --------------------------------------------------------
    # Family recall on validation
    # --------------------------------------------------------

    attack_predictions = (
        attack_scores < threshold
    )

    family_results = {}

    labels = validation_attack["label"]

    for name, attack_types in [
        ("DoS", DOS),
        ("Probe", PROBE),
        ("R2L", R2L),
        ("U2R", U2R),
    ]:

        mask = labels.isin(
            attack_types
        ).to_numpy()

        if mask.sum() > 0:
            family_results[name] = (
                attack_predictions[mask].sum()
                / mask.sum()
            )
        else:
            family_results[name] = 0.0

    result = {
        "feature_set": feature_name,
        "raw_features": len(feature_list),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc,
        "pr_auc": pr,
        "fpr": fpr,
        "DoS": family_results["DoS"],
        "Probe": family_results["Probe"],
        "R2L": family_results["R2L"],
        "U2R": family_results["U2R"],
    }

    print(f"Validation F1       : {f1:.4f}")
    print(f"Validation Recall   : {recall:.4f}")
    print(f"Validation Precision: {precision:.4f}")
    print(f"Validation PR-AUC   : {pr:.4f}")
    print(f"Validation ROC-AUC  : {roc:.4f}")
    print(f"Validation FPR      : {fpr:.4f}")

    print("\nFamily recall:")

    for family in [
        "DoS",
        "Probe",
        "R2L",
        "U2R"
    ]:
        print(
            f"  {family}: "
            f"{family_results[family]:.4f}"
        )

    print(
        f"\nTime: {time.time() - start:.1f}s"
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    if use_test:

        X_test = preprocessor.transform(
            test_df[feature_list]
        )

        test_scores = model.decision_function(
            X_test
        )

        test_pred = (
            test_scores < threshold
        ).astype(int)

        y_test = (
            test_df["label"] != "normal"
        ).astype(int).to_numpy()

        test_precision = precision_score(
            y_test,
            test_pred,
            zero_division=0
        )

        test_recall = recall_score(
            y_test,
            test_pred,
            zero_division=0
        )

        test_f1 = f1_score(
            y_test,
            test_pred,
            zero_division=0
        )

        test_roc = roc_auc_score(
            y_test,
            -test_scores
        )

        test_pr = average_precision_score(
            y_test,
            -test_scores
        )

        cm = confusion_matrix(
            y_test,
            test_pred
        )

        tn, fp, fn, tp = cm.ravel()

        test_fpr = fp / (fp + tn)

        print("\n" + "-" * 100)
        print("UNTOUCHED KDDTest+")
        print("-" * 100)

        print(
            f"Precision : {test_precision:.4f}"
        )
        print(
            f"Recall    : {test_recall:.4f}"
        )
        print(
            f"F1        : {test_f1:.4f}"
        )
        print(
            f"PR-AUC    : {test_pr:.4f}"
        )
        print(
            f"ROC-AUC   : {test_roc:.4f}"
        )
        print(
            f"Actual FPR: {test_fpr:.4f}"
        )

        attack_labels = test_df["label"]

        print("\nTest family recall:")

        for name, attack_types in [
            ("DoS", DOS),
            ("Probe", PROBE),
            ("R2L", R2L),
            ("U2R", U2R),
        ]:

            mask = attack_labels.isin(
                attack_types
            ).to_numpy()

            detected = test_pred[mask].sum()

            recall_family = (
                detected / mask.sum()
                if mask.sum() > 0
                else 0
            )

            print(
                f"  {name}: "
                f"{recall_family:.4f}"
            )

        unseen_types = {
            "apache2", "httptunnel", "mailbomb",
            "mscan", "named", "processtable",
            "ps", "sendmail", "snmpgetattack",
            "snmpguess", "sqlattack", "udpstorm",
            "worm", "xlock", "xsnoop"
        }

        unseen_mask = attack_labels.isin(
            unseen_types
        ).to_numpy()

        unseen_recall = (
            test_pred[unseen_mask].sum()
            / unseen_mask.sum()
        )

        print(
            f"  Unseen attacks: "
            f"{unseen_recall:.4f}"
        )

        result.update({
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "test_pr_auc": test_pr,
            "test_roc_auc": test_roc,
            "test_fpr": test_fpr,
            "test_unseen_recall": unseen_recall
        })

    return result


# ============================================================
# RUN ALL FEATURE SETS
# ============================================================

results = []

for name, features in FEATURE_SETS.items():

    result = evaluate_model(
        name,
        features,
        use_test=False
    )

    results.append(result)


# ============================================================
# FIND BEST CANDIDATES
# ============================================================

results_df = pd.DataFrame(results)

print("\n\n" + "=" * 100)
print("VALIDATION COMPARISON")
print("=" * 100)

print(
    results_df[
        [
            "feature_set",
            "raw_features",
            "f1",
            "recall",
            "precision",
            "pr_auc",
            "DoS",
            "Probe",
            "R2L",
            "U2R"
        ]
    ]
    .sort_values("f1", ascending=False)
    .to_string(index=False)
)


# ============================================================
# SELECT CANDIDATES
#
# We do NOT use KDDTest+ here.
#
# Pick:
# 1. best overall F1
# 2. best R2L/U2R-oriented candidate
# ============================================================

best_f1 = results_df.sort_values(
    "f1",
    ascending=False
).iloc[0]

results_df["minority_score"] = (
    results_df["R2L"]
    + results_df["U2R"]
)

best_minority = results_df.sort_values(
    "minority_score",
    ascending=False
).iloc[0]

candidate_names = list(dict.fromkeys([
    best_f1["feature_set"],
    best_minority["feature_set"],
    "baseline_41"
]))

print("\n" + "=" * 100)
print("CANDIDATES FOR FINAL TEST")
print("=" * 100)

for name in candidate_names:
    print(f"  {name}")


# ============================================================
# FINAL TEST ONLY FOR SELECTED CANDIDATES
# ============================================================

final_results = []

for name in candidate_names:

    result = evaluate_model(
        name,
        FEATURE_SETS[name],
        use_test=True
    )

    final_results.append(result)


# ============================================================
# FINAL COMPARISON
# ============================================================

final_df = pd.DataFrame(
    final_results
)

print("\n\n" + "=" * 100)
print("FINAL KDDTest+ COMPARISON")
print("=" * 100)

print(
    final_df[
        [
            "feature_set",
            "test_f1",
            "test_recall",
            "test_precision",
            "test_pr_auc",
            "test_roc_auc",
            "test_fpr",
            "test_unseen_recall"
        ]
    ]
    .sort_values(
        "test_f1",
        ascending=False
    )
    .to_string(index=False)
)


# ============================================================
# SAVE REPORT
# ============================================================

Path("reports").mkdir(
    exist_ok=True
)

results_df.to_json(
    "reports/feature_subset_validation.json",
    orient="records",
    indent=2
)

final_df.to_json(
    "reports/feature_subset_final_test.json",
    orient="records",
    indent=2
)

print("\nSaved:")
print("reports/feature_subset_validation.json")
print("reports/feature_subset_final_test.json")