"""
app.py
------
AegisSOC ML Service — the Python microservice sitting between your
Spring Boot backend and the trained model, exactly as laid out in the
design doc's architecture:

    Next.js -> Spring Boot -> Python ML Service -> ML Models

Run locally:
    uvicorn app:app --host 0.0.0.0 --port 8000 --app-dir src

Then, e.g. from Spring Boot (or curl/Postman):
    POST http://localhost:8000/api/v1/analyze
    POST http://localhost:8000/api/v1/analyze/batch
    GET  http://localhost:8000/health
    GET  http://localhost:8000/model/info
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from inference import AnomalyDetector

app = FastAPI(
    title="AegisSOC ML Anomaly Detection Service",
    description="Semi-supervised network/security-event anomaly detector "
                "(Isolation Forest, trained on NSL-KDD).",
    version="1.0.0",
)

detector: Optional[AnomalyDetector] = None


@app.on_event("startup")
def _load_model():
    global detector
    detector = AnomalyDetector()


class SecurityEvent(BaseModel):
    """
    Mirrors the normalized security-event shape from the design doc
    (Section 5). All fields are optional — anything omitted falls back
    to a safe default so partial log sources still work — but the more
    fields you provide, the more accurate the score.
    """
    model_config = ConfigDict(extra="allow")

    duration: Optional[float] = 0
    protocol_type: Optional[str] = "tcp"
    service: Optional[str] = "other"
    flag: Optional[str] = "SF"
    src_bytes: Optional[float] = 0
    dst_bytes: Optional[float] = 0
    land: Optional[int] = 0
    wrong_fragment: Optional[int] = 0
    urgent: Optional[int] = 0
    hot: Optional[int] = 0
    num_failed_logins: Optional[int] = 0
    logged_in: Optional[int] = 0
    num_compromised: Optional[int] = 0
    root_shell: Optional[int] = 0
    su_attempted: Optional[int] = 0
    num_root: Optional[int] = 0
    num_file_creations: Optional[int] = 0
    num_shells: Optional[int] = 0
    num_access_files: Optional[int] = 0
    num_outbound_cmds: Optional[int] = 0
    is_host_login: Optional[int] = 0
    is_guest_login: Optional[int] = 0
    count: Optional[float] = 1
    srv_count: Optional[float] = 1
    serror_rate: Optional[float] = 0
    srv_serror_rate: Optional[float] = 0
    rerror_rate: Optional[float] = 0
    srv_rerror_rate: Optional[float] = 0
    same_srv_rate: Optional[float] = 1
    diff_srv_rate: Optional[float] = 0
    srv_diff_host_rate: Optional[float] = 0
    dst_host_count: Optional[float] = 1
    dst_host_srv_count: Optional[float] = 1
    dst_host_same_srv_rate: Optional[float] = 1
    dst_host_diff_srv_rate: Optional[float] = 0
    dst_host_same_src_port_rate: Optional[float] = 0
    dst_host_srv_diff_host_rate: Optional[float] = 0
    dst_host_serror_rate: Optional[float] = 0
    dst_host_srv_serror_rate: Optional[float] = 0
    dst_host_rerror_rate: Optional[float] = 0
    dst_host_srv_rerror_rate: Optional[float] = 0

    # Free-form identifiers, not used by the model but useful to echo back
    # to the caller / attach to the resulting incident.
    source_ip: Optional[str] = None
    destination: Optional[str] = None
    username: Optional[str] = None
    event_id: Optional[str] = None


class AnalyzeResponse(BaseModel):
    event_id: Optional[str] = None
    is_anomaly: bool
    risk_score: float = Field(..., description="0-100 SOC risk score, higher = more anomalous")
    risk_level: str = Field(..., description="Low / Medium / High / Critical")
    raw_isolation_forest_score: float
    reasons: list[dict[str, Any]]


class BatchAnalyzeRequest(BaseModel):
    events: list[SecurityEvent]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": detector is not None}


@app.get("/model/info")
def model_info():
    if detector is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "model_type": detector.metadata["model_type"],
        "training_mode": detector.metadata["sklearn_training"],
        "dataset": detector.metadata["dataset"],
        "operating_threshold": detector.metadata["operating_threshold"],
        "target_false_positive_rate": detector.metadata["target_fpr"],
        "test_metrics": detector.metadata["test_metrics"],
    }


@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
def analyze(event: SecurityEvent):
    if detector is None:
        raise HTTPException(503, "Model not loaded")
    payload = event.model_dump()
    event_id = payload.pop("event_id", None)
    result = detector.predict_one(payload, explain=True)
    return {"event_id": event_id, **result}


@app.post("/api/v1/analyze/batch", response_model=list[AnalyzeResponse])
def analyze_batch(request: BatchAnalyzeRequest):
    if detector is None:
        raise HTTPException(503, "Model not loaded")
    payloads = []
    event_ids = []
    for e in request.events:
        d = e.model_dump()
        event_ids.append(d.pop("event_id", None))
        payloads.append(d)
    results = detector.predict_batch(payloads, explain=True)
    return [{"event_id": eid, **r} for eid, r in zip(event_ids, results)]
