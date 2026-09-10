"""
inference.py
------------
The reusable prediction interface. This is the module your fullstack
backend (or the FastAPI wrapper in app.py) calls to score a security
event. Loads the artifacts produced by train_model.py exactly once and
exposes a simple, well-typed function.

Usage:
    from inference import AnomalyDetector
    detector = AnomalyDetector()
    result = detector.predict_one({
        "duration": 0, "protocol_type": "tcp", "service": "private",
        "flag": "S0", "src_bytes": 0, "dst_bytes": 0, ...
    })
    # -> {"is_anomaly": True, "risk_score": 87.4, "risk_level": "Critical",
    #     "raw_score": -0.041, "reasons": [...]}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from preprocessing import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from risk_scoring import risk_bucket, scores_to_risk

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

# Sensible defaults so a caller can submit a partial event (e.g. only the
# fields their log source actually has) without crashing.
_DEFAULTS = {**{c: 0 for c in NUMERIC_FEATURES}, **{c: "unknown" for c in CATEGORICAL_FEATURES}}
_DEFAULTS.update({"protocol_type": "tcp", "service": "other", "flag": "SF", "logged_in": 0})


class AnomalyDetector:
    def __init__(self, model_dir: Path | str = MODEL_DIR):
        model_dir = Path(model_dir)
        self.preprocessor = joblib.load(model_dir / "preprocessor.joblib")
        self.model = joblib.load(model_dir / "isolation_forest.joblib")
        self.reason_explainer = joblib.load(model_dir / "reason_explainer.joblib")
        with open(model_dir / "metadata.json") as f:
            self.metadata = json.load(f)
        self.threshold = self.metadata["operating_threshold"]
        self.score_low = self.metadata["risk_score_calibration"]["score_low"]
        self.score_high = self.metadata["risk_score_calibration"]["score_high"]

    def _to_row(self, event: dict[str, Any]) -> pd.DataFrame:
        merged = {**_DEFAULTS, **event}
        row = {col: merged.get(col, _DEFAULTS.get(col, 0)) for col in ALL_FEATURES}
        return pd.DataFrame([row])

    def predict_one(self, event: dict[str, Any], explain: bool = True) -> dict[str, Any]:
        row_df = self._to_row(event)
        X = self.preprocessor.transform(row_df)
        raw_score = float(self.model.decision_function(X)[0])
        is_anomaly = raw_score < self.threshold
        risk_score = float(scores_to_risk(
            [raw_score], self.score_low, self.score_high
        )[0])

        result = {
            "is_anomaly": bool(is_anomaly),
            "risk_score": round(risk_score, 2),
            "risk_level": risk_bucket(risk_score),
            "raw_isolation_forest_score": round(raw_score, 5),
            "operating_threshold": round(self.threshold, 5),
        }
        if explain:
            result["reasons"] = self.reason_explainer.explain(merged_event(event))
        return result

    def predict_batch(self, events: list[dict[str, Any]], explain: bool = False) -> list[dict[str, Any]]:
        rows_df = pd.concat([self._to_row(e) for e in events], ignore_index=True)
        X = self.preprocessor.transform(rows_df)
        raw_scores = self.model.decision_function(X)
        risk_scores = scores_to_risk(raw_scores, self.score_low, self.score_high)

        results = []
        for i, event in enumerate(events):
            is_anomaly = bool(raw_scores[i] < self.threshold)
            r = {
                "is_anomaly": is_anomaly,
                "risk_score": round(float(risk_scores[i]), 2),
                "risk_level": risk_bucket(float(risk_scores[i])),
                "raw_isolation_forest_score": round(float(raw_scores[i]), 5),
            }
            if explain:
                r["reasons"] = self.reason_explainer.explain(merged_event(event))
            results.append(r)
        return results


def merged_event(event: dict[str, Any]) -> dict[str, Any]:
    return {**_DEFAULTS, **event}


if __name__ == "__main__":
    # Quick smoke test with a hand-crafted "obvious brute force" style event
    # and a hand-crafted "looks totally normal" event.
    detector = AnomalyDetector()

    suspicious_event = {
        "duration": 0, "protocol_type": "tcp", "service": "private", "flag": "S0",
        "src_bytes": 0, "dst_bytes": 0, "count": 123, "srv_count": 6,
        "serror_rate": 1.0, "srv_serror_rate": 1.0, "same_srv_rate": 0.05,
        "dst_host_count": 255, "dst_host_srv_count": 26, "num_failed_logins": 5,
    }
    normal_event = {
        "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
        "src_bytes": 232, "dst_bytes": 8153, "count": 1, "srv_count": 1,
        "logged_in": 1, "same_srv_rate": 1.0, "dst_host_count": 1, "dst_host_srv_count": 1,
    }

    print("=== Suspicious-looking event ===")
    print(json.dumps(detector.predict_one(suspicious_event), indent=2))
    print("\n=== Normal-looking event ===")
    print(json.dumps(detector.predict_one(normal_event), indent=2))
