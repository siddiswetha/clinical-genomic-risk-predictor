"""Synthetic patient data generation.

All data is synthetic, generated with a fixed seed by
``scripts/generate_data.py``. There are no real patients, no real
genomes, and no real clinical records in this repository. The
generation rules are documented in ``data/DATA_CARD.md`` and are
deliberately simple; they do NOT reflect real biology.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
N_SNP = 20

CLINICAL_FEATURES = [
    "age", "sex", "bmi", "sbp", "dbp", "total_cholesterol", "hdl",
    "fasting_glucose", "egfr", "smoking", "family_history",
]
SNP_FEATURES = [f"snp_{i + 1:02d}" for i in range(N_SNP)]
ALL_FEATURES = CLINICAL_FEATURES + SNP_FEATURES + ["prs"]

# Fixed SNP weights used to build the polygenic risk score (PRS).
# snp_03 and snp_07 are also the two drug-response markers.
SNP_WEIGHTS = np.array([
    0.18, -0.12, 0.35, 0.10, -0.08, 0.14, 0.30, -0.15, 0.11, 0.06,
    -0.09, 0.13, -0.05, 0.09, 0.07, -0.11, 0.05, 0.08, -0.06, 0.04,
])
SNP_MAF = np.array([
    0.30, 0.22, 0.18, 0.35, 0.28, 0.15, 0.20, 0.32, 0.25, 0.40,
    0.19, 0.27, 0.33, 0.21, 0.38, 0.24, 0.16, 0.29, 0.36, 0.23,
])


def generate_patients(n: int = 5000, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(55, 12, n), 30, 80)
    sex = rng.integers(0, 2, n)
    bmi = np.clip(rng.normal(27, 5, n), 18, 42)
    sbp = np.clip(120 + 0.6 * (age - 55) + 0.8 * (bmi - 27) + rng.normal(0, 12, n), 100, 180)
    dbp = np.clip(75 + 0.3 * (bmi - 27) + rng.normal(0, 8, n), 60, 110)
    total_cholesterol = np.clip(rng.normal(200, 35, n), 150, 280)
    hdl = np.clip(rng.normal(55, 12, n), 30, 90)
    fasting_glucose = np.clip(95 + 0.8 * (bmi - 27) + rng.normal(0, 15, n), 70, 180)
    egfr = np.clip(95 - 0.5 * (age - 55) + rng.normal(0, 12, n), 30, 120)
    smoking = (rng.random(n) < 0.25).astype(int)
    family_history = (rng.random(n) < 0.30).astype(int)

    snps = np.column_stack([
        rng.binomial(2, maf, n) for maf in SNP_MAF
    ]).astype(int)
    prs_raw = snps @ SNP_WEIGHTS
    prs = (prs_raw - prs_raw.mean()) / prs_raw.std()

    hba1c = np.clip(
        5.2 + 0.015 * (bmi - 25) + 0.008 * (age - 50)
        + 0.012 * (fasting_glucose - 95) + 0.15 * prs
        + rng.normal(0, 0.35, n),
        4.5, 11.0,
    )
    logit_risk = (
        -1.6 + 0.04 * (age - 55) + 0.08 * (bmi - 27)
        + 0.03 * (sbp - 130) / 10 + 0.02 * (fasting_glucose - 100) / 10
        + 0.50 * (hba1c - 5.7) + 0.40 * smoking + 0.30 * family_history
        + 0.60 * prs
    )
    p_risk = 1 / (1 + np.exp(-logit_risk))
    risk = (rng.random(n) < p_risk).astype(int)

    # Pharmacogenomics: responder to a synthetic anti-diabetic drug.
    # Deliberately driven partly by snp_03 and snp_07.
    logit_drug = (
        -0.5 + 0.90 * snps[:, 2] + 0.80 * snps[:, 6]
        - 0.05 * (bmi - 27) + 0.20 * (1 - smoking)
        + rng.normal(0, 0.5, n)
    )
    p_drug = 1 / (1 + np.exp(-logit_drug))
    drug_responder = (rng.random(n) < p_drug).astype(int)

    df = pd.DataFrame({
        "patient_id": [f"SYN-{i + 1:05d}" for i in range(n)],
        "age": np.round(age, 1),
        "sex": sex,
        "bmi": np.round(bmi, 1),
        "sbp": np.round(sbp, 1),
        "dbp": np.round(dbp, 1),
        "total_cholesterol": np.round(total_cholesterol, 1),
        "hdl": np.round(hdl, 1),
        "fasting_glucose": np.round(fasting_glucose, 1),
        "egfr": np.round(egfr, 1),
        "smoking": smoking,
        "family_history": family_history,
        "hba1c": np.round(hba1c, 2),
        "risk_5yr": risk,
        "drug_responder": drug_responder,
    })
    for j, name in enumerate(SNP_FEATURES):
        df[name] = snps[:, j]
    df["prs"] = np.round(prs, 3)
    return df
