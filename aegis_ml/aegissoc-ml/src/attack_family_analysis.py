import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent))

from preprocessing import build_preprocessor


RANDOM_STATE = 42

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

DATA = Path("data")

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

normal = train_df[train_df["label"] == "normal"].copy()

fit_normal, calib_normal = train_test_split(
    normal,
    test_size=0.25,
    random_state=RANDOM_STATE
)

preprocessor = build_preprocessor()

X_fit = preprocessor.fit_transform(fit_normal)
X_calib = preprocessor.transform(calib_normal)
X_test = preprocessor.transform(test_df)


# Current champion model
model = IsolationForest(
    n_estimators=800,
    max_samples=1.0,
    max_features=0.7,
    contamination=0.01,
    bootstrap=False,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(X_fit)

calib_scores = model.decision_function(X_calib)
test_scores = model.decision_function(X_test)

# Current conservative operating threshold
threshold = np.quantile(calib_scores, 0.01)

predicted = test_scores < threshold

test_df["score"] = test_scores
test_df["predicted_attack"] = predicted


# ---------------------------------------------------------
# Attack family mapping
# ---------------------------------------------------------

def family(label):
    if label == "normal":
        return "normal"

    dos = {
        "back", "land", "neptune", "pod", "smurf",
        "teardrop", "apache2", "udpstorm", "processtable",
        "mailbomb", "worm"
    }

    probe = {
        "ipsweep", "nmap", "portsweep", "satan",
        "mscan", "saint"
    }

    r2l = {
        "ftp_write", "guess_passwd", "imap", "multihop",
        "phf", "spy", "warezclient", "warezmaster",
        "named", "sendmail", "snmpgetattack", "snmpguess",
        "xlock", "xsnoop", "httptunnel"
    }

    u2r = {
        "buffer_overflow", "loadmodule", "perl",
        "rootkit", "ps", "sqlattack", "xterm"
    }

    if label in dos:
        return "DoS"
    if label in probe:
        return "Probe"
    if label in r2l:
        return "R2L"
    if label in u2r:
        return "U2R"

    return "unknown"


test_df["family"] = test_df["label"].apply(family)


# ---------------------------------------------------------
# Family statistics
# ---------------------------------------------------------

print("\n" + "=" * 95)
print("ATTACK FAMILY DIAGNOSTICS")
print("=" * 95)

for fam in ["DoS", "Probe", "R2L", "U2R"]:

    subset = test_df[test_df["family"] == fam]

    total = len(subset)
    detected = subset["predicted_attack"].sum()
    recall = detected / total if total else 0

    print(
        f"{fam:8s} | "
        f"Total: {total:6d} | "
        f"Detected: {detected:6d} | "
        f"Missed: {total-detected:6d} | "
        f"Recall: {recall:.4f}"
    )


# ---------------------------------------------------------
# Worst individual attack types
# ---------------------------------------------------------

print("\n" + "=" * 95)
print("WORST ATTACK TYPES")
print("=" * 95)

rows = []

for label, group in test_df[test_df["label"] != "normal"].groupby("label"):

    total = len(group)
    detected = group["predicted_attack"].sum()
    recall = detected / total

    rows.append({
        "attack": label,
        "family": group["family"].iloc[0],
        "total": total,
        "detected": detected,
        "missed": total - detected,
        "recall": recall,
        "mean_score": group["score"].mean(),
        "median_score": group["score"].median(),
    })

attack_stats = pd.DataFrame(rows)

print(
    attack_stats
    .sort_values("recall")
    .head(20)
    .to_string(index=False)
)


# ---------------------------------------------------------
# Score comparison
# ---------------------------------------------------------

print("\n" + "=" * 95)
print("SCORE DISTRIBUTION BY FAMILY")
print("=" * 95)

for fam in ["DoS", "Probe", "R2L", "U2R"]:

    scores = test_df[
        test_df["family"] == fam
    ]["score"]

    print(
        f"{fam:8s} | "
        f"mean={scores.mean():.5f} | "
        f"median={scores.median():.5f} | "
        f"min={scores.min():.5f} | "
        f"max={scores.max():.5f}"
    )


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

Path("reports").mkdir(exist_ok=True)

attack_stats.to_json(
    "reports/attack_family_analysis.json",
    orient="records",
    indent=2
)

print("\nSaved:")
print("reports/attack_family_analysis.json")
print(f"\nOperating threshold used: {threshold:.6f}")
