"""
AegisSOC — Isolation Forest Hyperparameter Tuning

Purpose:
    Find a better Isolation Forest configuration without changing
    the baseline train_model.py or touching the final test set
    during model selection.

Selection data:
    - NORMAL training data -> model fitting only
    - Held-out NORMAL data -> threshold calibration
    - Held-out ATTACK data -> model selection/evaluation only

Final KDDTest+ is NOT used to choose parameters.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from preprocessing import build_preprocessor, load_dataset


# ---------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"

REPORT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

# We keep the same target operating point as the baseline.
TARGET_FPR = 0.05


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def evaluate_at_threshold(scores, y_true, threshold):
    """
    decision_function:
        higher = more normal
        lower  = more anomalous

    Therefore:
        score < threshold => attack
    """

    y_pred = (scores < threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fpr": float(fpr),
        "flag_rate": float(y_pred.mean()),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


# ---------------------------------------------------------------------
# Attack-family metrics
# ---------------------------------------------------------------------

def family_recall(test_df, y_pred):
    results = {}

    for category in ["dos", "probe", "r2l", "u2r"]:
        mask = test_df["attack_category"] == category

        if mask.sum() > 0:
            results[f"{category}_recall"] = float(
                y_pred[mask.values].mean()
            )
        else:
            results[f"{category}_recall"] = None

    return results


def unseen_recall(train_df, test_df, y_pred):
    """
    Attack types that NEVER occur in the training file.
    """

    train_labels = set(train_df["label"].unique())

    novel_mask = (
        (test_df["is_anomaly"] == 1)
        & (~test_df["label"].isin(train_labels))
    )

    if novel_mask.sum() == 0:
        return None

    return float(y_pred[novel_mask.values].mean())


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    overall_start = time.time()

    print("=" * 80)
    print("AegisSOC — Isolation Forest Hyperparameter Tuning")
    print("=" * 80)

    # ---------------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------------

    train_df, test_df = load_dataset(
        str(DATA_DIR / "KDDTrain+.txt"),
        str(DATA_DIR / "KDDTest+.txt"),
    )

    print(f"\nTrain: {train_df.shape}")
    print(f"Test : {test_df.shape}")

    normal_df = train_df[
        train_df["is_anomaly"] == 0
    ].reset_index(drop=True)

    attack_df = train_df[
        train_df["is_anomaly"] == 1
    ].reset_index(drop=True)

    print(f"Normal rows: {len(normal_df)}")
    print(f"Attack rows: {len(attack_df)}")

    # ---------------------------------------------------------------
    # 2. Split NORMAL traffic
    #
    # Exactly the same 75/25 split used by train_model.py.
    # ---------------------------------------------------------------

    fit_normal_df, calib_normal_df = train_test_split(
        normal_df,
        test_size=0.25,
        random_state=RANDOM_STATE,
    )

    print("\nNormal split:")
    print(f"  Model fitting      : {len(fit_normal_df)}")
    print(f"  Threshold calib.   : {len(calib_normal_df)}")

    # ---------------------------------------------------------------
    # 3. Create validation attack set
    #
    # IMPORTANT:
    # Attack data is NEVER passed to model.fit().
    #
    # We use it only to compare already-trained models.
    # ---------------------------------------------------------------

    attack_validation_df = attack_df.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    validation_df = pd.concat(
        [
            calib_normal_df,
            attack_validation_df,
        ],
        ignore_index=True,
    )

    y_validation = validation_df["is_anomaly"].values

    print(f"\nValidation set:")
    print(f"  Normal : {len(calib_normal_df)}")
    print(f"  Attack : {len(attack_validation_df)}")
    print(f"  Total  : {len(validation_df)}")

    # ---------------------------------------------------------------
    # 4. Fit preprocessing ONLY on model-fitting normal data
    # ---------------------------------------------------------------

    preprocessor = build_preprocessor()

    X_fit = preprocessor.fit_transform(
        fit_normal_df
    )

    X_validation = preprocessor.transform(
        validation_df
    )

    print(
        f"\nFeature space: {X_fit.shape[1]} dimensions"
    )

    # ---------------------------------------------------------------
    # 5. Hyperparameter search
    # ---------------------------------------------------------------
    #
    # We deliberately keep the grid manageable.
    #
    # n_estimators:
    #   More trees generally make the forest more stable,
    #   but increase training time.
    #
    # max_samples:
    #   Controls how many samples each tree sees.
    #
    # max_features:
    #   Controls how many features each tree considers.
    #
    # contamination:
    #   Affects the learned offset used by Isolation Forest.
    #   We calibrate OUR operating threshold separately, but test
    #   a small range because it can still affect the model's
    #   decision-function scaling/behaviour.
    # ---------------------------------------------------------------

    param_grid = [
        {
            "n_estimators": n,
            "max_samples": ms,
            "max_features": mf,
            "contamination": contamination,
            "bootstrap": False,
        }
        for n in [300, 500, 800]
        for ms in ["auto", 0.5, 1.0]
        for mf in [0.7, 1.0]
        for contamination in [0.01, 0.05]
    ]

    print(
        f"\nTesting {len(param_grid)} configurations..."
    )

    results = []

    for i, params in enumerate(param_grid, start=1):

        start = time.time()

        model = IsolationForest(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            **params,
        )

        model.fit(X_fit)

        # -----------------------------------------------------------
        # Threshold calibration on NORMAL validation data only.
        # Exactly the same philosophy as train_model.py.
        # -----------------------------------------------------------

        calib_scores = model.decision_function(
            preprocessor.transform(calib_normal_df)
        )

        threshold = float(
            np.quantile(
                calib_scores,
                TARGET_FPR,
            )
        )

        # -----------------------------------------------------------
        # Evaluate on validation data containing attacks.
        # -----------------------------------------------------------

        validation_scores = model.decision_function(
            X_validation
        )

        metrics = evaluate_at_threshold(
            validation_scores,
            y_validation,
            threshold,
        )

        anomaly_likelihood = -validation_scores

        pr_auc = average_precision_score(
            y_validation,
            anomaly_likelihood,
        )

        roc_auc = roc_auc_score(
            y_validation,
            anomaly_likelihood,
        )

        elapsed = time.time() - start

        row = {
            **params,
            "threshold": threshold,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "fpr": metrics["fpr"],
            "pr_auc": float(pr_auc),
            "roc_auc": float(roc_auc),
            "training_seconds": elapsed,
        }

        results.append(row)

        print(
            f"[{i:02d}/{len(param_grid)}] "
            f"trees={params['n_estimators']} "
            f"samples={params['max_samples']} "
            f"features={params['max_features']} "
            f"contam={params['contamination']} "
            f"| F1={metrics['f1']:.4f} "
            f"R={metrics['recall']:.4f} "
            f"P={metrics['precision']:.4f} "
            f"PR-AUC={pr_auc:.4f} "
            f"| {elapsed:.2f}s"
        )

    # ---------------------------------------------------------------
    # 6. Sort candidates
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(results)

    # Primary objective: F1
    # Secondary: recall
    # Tertiary: PR-AUC
    results_df = results_df.sort_values(
        by=[
            "f1",
            "recall",
            "pr_auc",
        ],
        ascending=False,
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # 7. Display top candidates
    # ---------------------------------------------------------------

    print("\n" + "=" * 80)
    print("TOP 10 CONFIGURATIONS")
    print("=" * 80)

    display_columns = [
        "n_estimators",
        "max_samples",
        "max_features",
        "contamination",
        "f1",
        "recall",
        "precision",
        "pr_auc",
        "roc_auc",
        "fpr",
        "threshold",
        "training_seconds",
    ]

    print(
        results_df[
            display_columns
        ].head(10).to_string(index=False)
    )

    # ---------------------------------------------------------------
    # 8. Best candidate
    # ---------------------------------------------------------------

    best = results_df.iloc[0]

    print("\n" + "=" * 80)
    print("BEST VALIDATION CONFIGURATION")
    print("=" * 80)

    print(
        f"n_estimators  : {int(best['n_estimators'])}"
    )
    print(
        f"max_samples   : {best['max_samples']}"
    )
    print(
        f"max_features  : {best['max_features']}"
    )
    print(
        f"contamination  : {best['contamination']}"
    )
    print(
        f"threshold      : {best['threshold']:.6f}"
    )

    print("\nValidation performance:")
    print(
        f"Precision      : {best['precision']:.4f}"
    )
    print(
        f"Recall         : {best['recall']:.4f}"
    )
    print(
        f"F1             : {best['f1']:.4f}"
    )
    print(
        f"PR-AUC         : {best['pr_auc']:.4f}"
    )
    print(
        f"ROC-AUC        : {best['roc_auc']:.4f}"
    )
    print(
        f"FPR            : {best['fpr']:.4f}"
    )

    # ---------------------------------------------------------------
    # 9. Save tuning report
    # ---------------------------------------------------------------

    output_path = (
        REPORT_DIR / "isolation_forest_tuning.csv"
    )

    results_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"\nSaved tuning results to:\n{output_path}"
    )

    # ---------------------------------------------------------------
    # 10. Train BEST model and evaluate on untouched KDDTest+
    #
    # This happens ONLY after the configuration has been selected.
    # ---------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FINAL EVALUATION ON UNTOUCHED KDDTest+")
    print("=" * 80)

    best_model = IsolationForest(
        n_estimators=int(best["n_estimators"]),
        max_samples=best["max_samples"],
        max_features=best["max_features"],
        contamination=best["contamination"],
        bootstrap=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    final_start = time.time()

    best_model.fit(X_fit)

    # Threshold remains calibrated using held-out normal traffic.
    calib_scores = best_model.decision_function(
        preprocessor.transform(calib_normal_df)
    )

    final_threshold = float(
        np.quantile(
            calib_scores,
            TARGET_FPR,
        )
    )

    X_test = preprocessor.transform(test_df)

    test_scores = best_model.decision_function(
        X_test
    )

    y_test = test_df["is_anomaly"].values

    test_metrics = evaluate_at_threshold(
        test_scores,
        y_test,
        final_threshold,
    )

    anomaly_likelihood = -test_scores

    test_pr_auc = average_precision_score(
        y_test,
        anomaly_likelihood,
    )

    test_roc_auc = roc_auc_score(
        y_test,
        anomaly_likelihood,
    )

    y_test_pred = (
        test_scores < final_threshold
    ).astype(int)

    family_metrics = family_recall(
        test_df,
        y_test_pred,
    )

    novel_recall = unseen_recall(
        train_df,
        test_df,
        y_test_pred,
    )

    final_time = time.time() - final_start

    print(
        f"\nThreshold       : {final_threshold:.6f}"
    )
    print(
        f"Precision       : {test_metrics['precision']:.4f}"
    )
    print(
        f"Recall          : {test_metrics['recall']:.4f}"
    )
    print(
        f"F1              : {test_metrics['f1']:.4f}"
    )
    print(
        f"PR-AUC          : {test_pr_auc:.4f}"
    )
    print(
        f"ROC-AUC         : {test_roc_auc:.4f}"
    )
    print(
        f"Actual FPR      : {test_metrics['fpr']:.4f}"
    )

    print("\nAttack-family recall:")

    for key, value in family_metrics.items():
        print(
            f"  {key}: "
            f"{value:.4f}"
        )

    print(
        f"\nUnseen attack recall: "
        f"{novel_recall:.4f}"
    )

    print(
        f"\nFinal evaluation time: "
        f"{final_time:.2f}s"
    )

    # ---------------------------------------------------------------
    # 11. Compare against baseline
    # ---------------------------------------------------------------

    baseline = {
        "f1": 0.7695,
        "recall": 0.6542,
        "precision": 0.9342,
        "pr_auc": 0.9510,
        "roc_auc": 0.9378,
        "unseen_recall": 0.6573,
        "r2l_recall": 0.0659,
        "u2r_recall": 0.2687,
    }

    print("\n" + "=" * 80)
    print("BASELINE vs TUNED")
    print("=" * 80)

    comparisons = {
        "F1": (
            baseline["f1"],
            test_metrics["f1"],
        ),
        "Recall": (
            baseline["recall"],
            test_metrics["recall"],
        ),
        "Precision": (
            baseline["precision"],
            test_metrics["precision"],
        ),
        "PR-AUC": (
            baseline["pr_auc"],
            test_pr_auc,
        ),
        "ROC-AUC": (
            baseline["roc_auc"],
            test_roc_auc,
        ),
        "Unseen recall": (
            baseline["unseen_recall"],
            novel_recall,
        ),
        "R2L recall": (
            baseline["r2l_recall"],
            family_metrics["r2l_recall"],
        ),
        "U2R recall": (
            baseline["u2r_recall"],
            family_metrics["u2r_recall"],
        ),
    }

    for name, (old, new) in comparisons.items():

        delta = new - old

        print(
            f"{name:15s}: "
            f"{old:.4f} → {new:.4f} "
            f"({delta:+.4f})"
        )

    # ---------------------------------------------------------------
    # 12. Save final tuning summary
    # ---------------------------------------------------------------

    summary = {
        "best_parameters": {
            "n_estimators": int(best["n_estimators"]),
            "max_samples": best["max_samples"],
            "max_features": best["max_features"],
            "contamination": best["contamination"],
            "bootstrap": False,
        },
        "final_threshold": final_threshold,
        "baseline": baseline,
        "tuned": {
            "precision": test_metrics["precision"],
            "recall": test_metrics["recall"],
            "f1": test_metrics["f1"],
            "pr_auc": float(test_pr_auc),
            "roc_auc": float(test_roc_auc),
            "fpr": test_metrics["fpr"],
            "unseen_recall": novel_recall,
            **family_metrics,
        },
        "training_seconds": final_time,
    }

    import json

    summary_path = (
        REPORT_DIR /
        "isolation_forest_tuning_summary.json"
    )

    with open(summary_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        f"\nSaved final summary to:\n{summary_path}"
    )

    print(
        f"\nTotal tuning time: "
        f"{time.time() - overall_start:.1f}s"
    )

    print("\nDONE.")


if __name__ == "__main__":
    main()

    