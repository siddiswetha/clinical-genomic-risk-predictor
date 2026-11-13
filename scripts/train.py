"""Train all models and write evals/*.json with real measured numbers.

Run:  python scripts/train.py
Reads data/patients.csv (generate it first with scripts/generate_data.py).
Saves models to models/ and results to evals/classification.json,
evals/regression.json, evals/clustering.json, evals/ablation.json,
evals/pharmacogenomics.json.
"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from clinical_genomic.data import ALL_FEATURES, CLINICAL_FEATURES
from clinical_genomic.features import DRUG_MARKERS, GENOMIC_FEATURES
from clinical_genomic.models import train_keras_classifier, train_keras_regressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "patients.csv"
MODELS = ROOT / "models"
EVALS = ROOT / "evals"


def clf_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
    }


def main() -> None:
    MODELS.mkdir(exist_ok=True)
    EVALS.mkdir(exist_ok=True)
    df = pd.read_csv(DATA)
    print(f"rows={len(df)} risk prevalence={df['risk_5yr'].mean():.3f} "
          f"drug responder rate={df['drug_responder'].mean():.3f}")

    X = df[ALL_FEATURES]
    y = df["risk_5yr"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    # ---------- Classification ----------
    classifiers = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "decision_tree": DecisionTreeClassifier(max_depth=6, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=2),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    classification = {}
    best_name, best_auc = None, -1.0
    for name, model in classifiers.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=2)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = clf_metrics(y_test, pred, proba)
        m["cv_roc_auc_mean"] = round(float(scores.mean()), 4)
        m["cv_roc_auc_std"] = round(float(scores.std()), 4)
        classification[name] = m
        print(f"clf {name}: {m}")
        joblib.dump(model, MODELS / f"clf_{name}.joblib")
        if m["roc_auc"] > best_auc:
            best_name, best_auc = name, m["roc_auc"]

    # Keras MLP classifier (scaled features, internal val split for early stopping)
    scaler = StandardScaler().fit(X_train)
    Xtr_s = scaler.transform(X_train)
    Xte_s = scaler.transform(X_test)
    X_tr, X_val, y_tr, y_val = train_test_split(
        Xtr_s, y_train, test_size=0.15, random_state=42, stratify=y_train
    )
    keras_clf = train_keras_classifier(X_tr, y_tr.values, X_val, y_val.values)
    proba_k = keras_clf.predict(Xte_s, verbose=0).ravel()
    pred_k = (proba_k >= 0.5).astype(int)
    classification["keras_mlp"] = clf_metrics(y_test, pred_k, proba_k)
    classification["keras_mlp"]["cv_roc_auc_mean"] = None
    classification["keras_mlp"]["cv_roc_auc_std"] = None
    print(f"clf keras_mlp: {classification['keras_mlp']}")
    keras_clf.save(MODELS / "clf_keras_mlp.keras")
    joblib.dump(scaler, MODELS / "scaler.joblib")

    # Threshold analysis + permutation importance for the best classical model
    best_model = joblib.load(MODELS / f"clf_{best_name}.joblib")
    thresholds = {}
    proba_best = best_model.predict_proba(X_test)[:, 1]
    for t in (0.3, 0.4, 0.5, 0.6, 0.7):
        p = (proba_best >= t).astype(int)
        thresholds[str(t)] = {
            "precision": round(float(precision_score(y_test, p, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, p, zero_division=0)), 4),
            "f1": round(float(f1_score(y_test, p, zero_division=0)), 4),
        }
    perm = permutation_importance(
        best_model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=2
    )
    top_idx = np.argsort(perm.importances_mean)[::-1][:10]
    importance = [
        {"feature": ALL_FEATURES[i], "importance": round(float(perm.importances_mean[i]), 4)}
        for i in top_idx
    ]
    clf_out = {
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "target": "risk_5yr", "split": "80/20 stratified, seed 42",
        "models": classification,
        "best_model_by_heldout_roc_auc": best_name,
        "threshold_analysis_best_model": thresholds,
        "permutation_importance_best_model_top10": importance,
    }
    (EVALS / "classification.json").write_text(json.dumps(clf_out, indent=2))

    # ---------- Regression: predict HbA1c ----------
    y_reg = df["hba1c"]
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(
        X, y_reg, test_size=0.20, random_state=42
    )
    # HbA1c is itself generated from the features, so exclude it from inputs.
    reg_features = [f for f in ALL_FEATURES]
    regressors = {
        "ridge": Pipeline([("scaler", StandardScaler()), ("reg", Ridge(alpha=1.0))]),
        "random_forest_regressor": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=2),
    }
    regression = {}
    for name, model in regressors.items():
        model.fit(Xr_train[reg_features], yr_train)
        pred = model.predict(Xr_test[reg_features])
        regression[name] = {
            "rmse": round(float(np.sqrt(mean_squared_error(yr_test, pred))), 4),
            "mae": round(float(mean_absolute_error(yr_test, pred)), 4),
            "r2": round(float(r2_score(yr_test, pred)), 4),
        }
        print(f"reg {name}: {regression[name]}")
        joblib.dump(model, MODELS / f"reg_{name}.joblib")
    scaler_r = StandardScaler().fit(Xr_train[reg_features])
    Xrr = scaler_r.transform(Xr_train[reg_features])
    Xrt = scaler_r.transform(Xr_test[reg_features])
    X_tr, X_val, y_tr, y_val = train_test_split(Xrr, yr_train, test_size=0.15, random_state=42)
    keras_reg = train_keras_regressor(X_tr, y_tr.values, X_val, y_val.values)
    pred_k = keras_reg.predict(Xrt, verbose=0).ravel()
    regression["keras_mlp"] = {
        "rmse": round(float(np.sqrt(mean_squared_error(yr_test, pred_k))), 4),
        "mae": round(float(mean_absolute_error(yr_test, pred_k)), 4),
        "r2": round(float(r2_score(yr_test, pred_k)), 4),
    }
    print(f"reg keras_mlp: {regression['keras_mlp']}")
    keras_reg.save(MODELS / "reg_keras_mlp.keras")
    (EVALS / "regression.json").write_text(json.dumps({
        "n_train": int(len(Xr_train)), "n_test": int(len(Xr_test)),
        "target": "hba1c", "split": "80/20, seed 42",
        "note": "HbA1c is generated from the input features by construction; "
                "this measures recovery of the synthetic rule, not clinical prediction.",
        "models": regression,
    }, indent=2))

    # ---------- Clustering ----------
    scaler_c = StandardScaler().fit(X)
    X_scaled = scaler_c.transform(X)
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    df_c = df.copy()
    df_c["cluster"] = labels
    profile = []
    for c in range(4):
        sub = df_c[df_c["cluster"] == c]
        profile.append({
            "cluster": int(c), "n": int(len(sub)),
            "mean_age": round(float(sub["age"].mean()), 1),
            "mean_bmi": round(float(sub["bmi"].mean()), 1),
            "mean_hba1c": round(float(sub["hba1c"].mean()), 2),
            "mean_prs": round(float(sub["prs"].mean()), 3),
            "risk_prevalence": round(float(sub["risk_5yr"].mean()), 3),
        })
    clustering = {
        "method": "KMeans k=4 on scaled clinical+genomic features",
        "silhouette_score": round(float(sil), 4),
        "cluster_sizes": [int((labels == c).sum()) for c in range(4)],
        "profile": profile,
        "note": "Clusters are exploratory groupings of synthetic data; "
                "they are not real disease subtypes.",
    }
    print(f"clustering: silhouette={sil:.4f} sizes={clustering['cluster_sizes']}")
    (EVALS / "clustering.json").write_text(json.dumps(clustering, indent=2))

    # ---------- Genomics ablation (LogReg, held-out ROC-AUC) ----------
    ablation = {}
    for label, feats in [
        ("clinical_only", CLINICAL_FEATURES),
        ("clinical_plus_genomic", ALL_FEATURES),
        ("genomic_only", GENOMIC_FEATURES),
    ]:
        m = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=1000, random_state=42))])
        m.fit(X_train[feats], y_train)
        proba = m.predict_proba(X_test[feats])[:, 1]
        ablation[label] = round(float(roc_auc_score(y_test, proba)), 4)
    ablation["genomic_gain_clinical_plus_genomic_minus_clinical_only"] = round(
        ablation["clinical_plus_genomic"] - ablation["clinical_only"], 4)
    print(f"ablation: {ablation}")
    (EVALS / "ablation.json").write_text(json.dumps(ablation, indent=2))

    # ---------- Pharmacogenomics ----------
    pharm = {}
    y_drug = df["drug_responder"]
    Xd_train, Xd_test, yd_train, yd_test = train_test_split(
        X, y_drug, test_size=0.20, random_state=42, stratify=y_drug
    )
    for label, feats in [
        ("with_markers", ALL_FEATURES),
        # Without the markers AND without PRS: PRS is a weighted sum of
        # the SNPs including snp_03/snp_07, so leaving it in would leak
        # the marker signal and make the ablation meaningless.
        ("without_markers", [f for f in ALL_FEATURES if f not in DRUG_MARKERS + ["prs"]]),
    ]:
        m = Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=1000, random_state=42))])
        m.fit(Xd_train[feats], yd_train)
        proba = m.predict_proba(Xd_test[feats])[:, 1]
        pred = (proba >= 0.5).astype(int)
        pharm[label] = clf_metrics(yd_test, pred, proba)
    pharm["roc_auc_gain_with_markers"] = round(
        pharm["with_markers"]["roc_auc"] - pharm["without_markers"]["roc_auc"], 4)
    print(f"pharmaco: {pharm}")
    (EVALS / "pharmacogenomics.json").write_text(json.dumps(pharm, indent=2))
    print("done — evals written to evals/")


if __name__ == "__main__":
    main()
