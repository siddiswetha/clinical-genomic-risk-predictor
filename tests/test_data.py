import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clinical_genomic.data import ALL_FEATURES, generate_patients

ROOT = Path(__file__).resolve().parents[1]


def test_generate_shape_and_ids():
    df = generate_patients(n=500, seed=42)
    assert len(df) == 500
    assert df["patient_id"].str.startswith("SYN-").all()
    assert df["patient_id"].is_unique


def test_generate_deterministic():
    a = generate_patients(n=100, seed=42)
    b = generate_patients(n=100, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_feature_ranges():
    df = generate_patients(n=500, seed=42)
    assert df["age"].between(30, 80).all()
    assert df["bmi"].between(18, 42).all()
    assert df["hba1c"].between(4.5, 11.0).all()
    for c in [f"snp_{i + 1:02d}" for i in range(20)]:
        assert df[c].isin([0, 1, 2]).all()
    assert set(df["risk_5yr"].unique()) <= {0, 1}
    assert set(df["drug_responder"].unique()) <= {0, 1}


def test_all_features_present():
    df = generate_patients(n=50, seed=42)
    for f in ALL_FEATURES:
        assert f in df.columns


def test_csv_exists_and_has_no_real_ids():
    p = ROOT / "data" / "patients.csv"
    assert p.exists()
    df = pd.read_csv(p)
    assert len(df) == 5000
    assert not df["patient_id"].str.contains("^[0-9]+$").any()
