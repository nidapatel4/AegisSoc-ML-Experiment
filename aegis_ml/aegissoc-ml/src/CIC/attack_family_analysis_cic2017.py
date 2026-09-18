import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data/MachineLearningCVE"
MODEL_PATH = BASE_DIR / "models/CIC/cic2017_isolation_forest_baseline.joblib"

X_TEST_PATH = BASE_DIR / "models/CIC/X_test.npy"

REPORT_PATH = BASE_DIR / "reports/CIC/cic2017_attack_family_analysis.json"

# Baseline CIC operating point
THRESHOLD = 0.058963


TEST_FILES = [
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]


def clean_label(label):
    label = str(label).strip()

    if label.upper() == "BENIGN":
        return "BENIGN"

    return label


def classify_family(label):

    label_lower = label.lower()

    if label == "BENIGN":
        return "BENIGN"

    if "dos" in label_lower or "ddos" in label_lower:
        return "DoS/DDoS"

    if "portscan" in label_lower:
        return "PortScan"

    if "patator" in label_lower:
        return "BruteForce"

    if "web attack" in label_lower:
        return "WebAttack"

    if "infiltration" in label_lower:
        return "Infiltration"

    if "bot" in label_lower:
        return "Bot"

    if "heartbleed" in label_lower:
        return "Heartbleed"

    return "OtherAttack"


def main():

    print("Loading model and test data...")

    model = joblib.load(MODEL_PATH)
    X_test = np.load(X_TEST_PATH)

    print(f"X_test rows: {len(X_test):,}")

    print()
    print("Loading original CIC test labels...")

    labels = []

    for filename in TEST_FILES:

        path = DATA_DIR / filename

        print(f"Reading {filename}...")

        df = pd.read_csv(path)

        df.columns = (
            df.columns
            .astype(str)
            .str.strip()
        )

        file_labels = (
            df["Label"]
            .map(clean_label)
            .tolist()
        )

        labels.extend(file_labels)

    labels = np.array(labels)

    print(f"Labels loaded: {len(labels):,}")

    if len(labels) != len(X_test):
        raise ValueError(
            f"Mismatch!\n"
            f"X_test rows = {len(X_test):,}\n"
            f"labels     = {len(labels):,}"
        )

    print()
    print("Generating baseline predictions...")

    scores = model.decision_function(X_test)

    # Lower Isolation Forest score = more anomalous
    y_pred = (scores < THRESHOLD).astype(int)

    y_true = (labels != "BENIGN").astype(int)

    overall_cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = overall_cm.ravel()

    print()
    print("=" * 70)
    print("OVERALL")
    print("=" * 70)

    print(f"TN: {tn:,}")
    print(f"FP: {fp:,}")
    print(f"FN: {fn:,}")
    print(f"TP: {tp:,}")

    print()
    print("Attack-family analysis:")

    families = np.array([
        classify_family(label)
        for label in labels
    ])

    results = {}

    attack_families = sorted(
        set(families) - {"BENIGN"}
    )

    for family in attack_families:

        mask = families == family

        total = int(mask.sum())

        detected = int(
            (y_pred[mask] == 1).sum()
        )

        missed = total - detected

        recall = (
            detected / total
            if total > 0
            else 0.0
        )

        results[family] = {
            "total_attacks": total,
            "detected": detected,
            "missed": missed,
            "recall": float(recall),
        }

        print(
            f"{family:15s} | "
            f"Total: {total:8,d} | "
            f"Detected: {detected:8,d} | "
            f"Missed: {missed:8,d} | "
            f"Recall: {recall * 100:6.2f}%"
        )

    # Also preserve exact original CIC labels
    exact_label_results = {}

    attack_labels = sorted(
        set(labels) - {"BENIGN"}
    )

    for label in attack_labels:

        mask = labels == label

        total = int(mask.sum())

        detected = int(
            (y_pred[mask] == 1).sum()
        )

        recall = (
            detected / total
            if total > 0
            else 0.0
        )

        exact_label_results[label] = {
            "total": total,
            "detected": detected,
            "missed": total - detected,
            "recall": float(recall),
        }

    report = {
        "experiment": (
            "CIC-IDS2017 attack-family analysis"
        ),

        "model": "CIC Isolation Forest baseline",

        "threshold": THRESHOLD,

        "test_files": TEST_FILES,

        "test_rows": int(len(X_test)),

        "overall_confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },

        "family_results": results,

        "exact_attack_label_results": exact_label_results,
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    print()
    print("=" * 70)
    print("ATTACK-FAMILY ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"Report saved to:")
    print(REPORT_PATH)


if __name__ == "__main__":
    main()