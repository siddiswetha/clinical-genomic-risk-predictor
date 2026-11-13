"""FastAPI service for the clinical-genomic risk predictor.

Research / educational prototype only. Not medical advice, not for
clinical use, no real patients. All data is synthetic.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .data import ALL_FEATURES

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
RUNS = ROOT / "runs"

logging.basicConfig(level=logging.INFO, format='{"time": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}')
log = logging.getLogger(__name__)
DISCLAIMER = (
    "Research / educational prototype only. Synthetic data. "
    "Not medical advice and not for clinical use."
)

app = FastAPI(title="clinical-genomic-risk-predictor", version="0.1.0")

_models: dict = {}


def _load_models():
    if _models:
        return _models
    _models["risk"] = joblib.load(MODELS / "clf_random_forest.joblib")
    _models["hba1c"] = joblib.load(MODELS / "reg_random_forest_regressor.joblib")
    return _models


class PatientInput(BaseModel):
    age: float = Field(..., ge=30, le=80)
    sex: int = Field(..., ge=0, le=1)
    bmi: float = Field(..., ge=18, le=42)
    sbp: float = Field(..., ge=100, le=180)
    dbp: float = Field(..., ge=60, le=110)
    total_cholesterol: float = Field(..., ge=150, le=280)
    hdl: float = Field(..., ge=30, le=90)
    fasting_glucose: float = Field(..., ge=70, le=180)
    egfr: float = Field(..., ge=30, le=120)
    smoking: int = Field(..., ge=0, le=1)
    family_history: int = Field(..., ge=0, le=1)
    prs: float = Field(0.0, ge=-3, le=3)
    # SNPs default to 0 (major-allele homozygous) if not supplied.
    snp_01: int = 0
    snp_02: int = 0
    snp_03: int = 0
    snp_04: int = 0
    snp_05: int = 0
    snp_06: int = 0
    snp_07: int = 0
    snp_08: int = 0
    snp_09: int = 0
    snp_10: int = 0
    snp_11: int = 0
    snp_12: int = 0
    snp_13: int = 0
    snp_14: int = 0
    snp_15: int = 0
    snp_16: int = 0
    snp_17: int = 0
    snp_18: int = 0
    snp_19: int = 0
    snp_20: int = 0


def _row(payload: PatientInput) -> pd.DataFrame:
    data = {f: getattr(payload, f) for f in ALL_FEATURES}
    return pd.DataFrame([data])


def _save_run(trace_id: str, payload: dict) -> None:
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{trace_id}.json").write_text(json.dumps(payload, indent=2))


@app.get("/health")
def health():
    return {"status": "ok", "disclaimer": DISCLAIMER}


@app.get("/metrics")
def metrics():
    return {
        "service": "clinical-genomic-risk-predictor",
        "models_loaded": bool(_models),
        "disclaimer": DISCLAIMER,
    }


@app.post("/predict-risk")
def predict_risk(payload: PatientInput):
    trace_id = str(uuid.uuid4())
    models = _load_models()
    row = _row(payload)
    proba = float(models["risk"].predict_proba(row)[0, 1])
    result = {
        "trace_id": trace_id,
        "risk_5yr_probability": round(proba, 4),
        "risk_5yr_prediction": int(proba >= 0.5),
        "disclaimer": DISCLAIMER,
    }
    _save_run(trace_id, {"endpoint": "predict-risk", **result})
    log.info("predict-risk trace_id=%s prob=%.4f", trace_id, proba)
    return result


@app.post("/predict-hba1c")
def predict_hba1c(payload: PatientInput):
    trace_id = str(uuid.uuid4())
    models = _load_models()
    row = _row(payload)
    value = float(models["hba1c"].predict(row)[0])
    result = {
        "trace_id": trace_id,
        "predicted_hba1c": round(value, 2),
        "disclaimer": DISCLAIMER,
    }
    _save_run(trace_id, {"endpoint": "predict-hba1c", **result})
    log.info("predict-hba1c trace_id=%s value=%.2f", trace_id, value)
    return result
