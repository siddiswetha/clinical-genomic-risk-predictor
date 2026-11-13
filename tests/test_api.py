import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient

from clinical_genomic.app import app

client = TestClient(app)

PAYLOAD = {
    "age": 55, "sex": 0, "bmi": 27.0, "sbp": 130, "dbp": 80,
    "total_cholesterol": 200, "hdl": 55, "fasting_glucose": 95,
    "egfr": 90, "smoking": 0, "family_history": 0, "prs": 0.0,
}


def test_health_has_disclaimer():
    r = client.get("/health")
    assert r.status_code == 200
    assert "not for clinical use" in r.json()["disclaimer"].lower()


def test_metrics():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.json()["service"] == "clinical-genomic-risk-predictor"


def test_predict_risk():
    r = client.post("/predict-risk", json=PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk_5yr_probability"] <= 1.0
    assert "disclaimer" in body
    assert body["risk_5yr_prediction"] in (0, 1)


def test_predict_hba1c():
    r = client.post("/predict-hba1c", json=PAYLOAD)
    assert r.status_code == 200
    assert 3.0 <= r.json()["predicted_hba1c"] <= 12.0


def test_invalid_input_rejected():
    bad = dict(PAYLOAD, age=5)
    r = client.post("/predict-risk", json=bad)
    assert r.status_code == 422


def test_readme_disclaimer():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    assert "synthetic" in readme.lower()
    assert "not medical advice" in readme.lower()
