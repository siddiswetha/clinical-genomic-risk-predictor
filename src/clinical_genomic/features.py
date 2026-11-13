"""Feature lists and preprocessing helpers."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data import ALL_FEATURES, CLINICAL_FEATURES, SNP_FEATURES

GENOMIC_FEATURES = SNP_FEATURES + ["prs"]
DRUG_MARKERS = ["snp_03", "snp_07"]


def clinical_only_features() -> list[str]:
    return list(CLINICAL_FEATURES)


def genomic_only_features() -> list[str]:
    return list(GENOMIC_FEATURES)


def all_features() -> list[str]:
    return list(ALL_FEATURES)


def scaler_pipeline() -> Pipeline:
    """Median imputation + standard scaling for all numeric features."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def make_preprocessor(feature_names: list[str]) -> ColumnTransformer:
    return ColumnTransformer([("num", scaler_pipeline(), feature_names)])
