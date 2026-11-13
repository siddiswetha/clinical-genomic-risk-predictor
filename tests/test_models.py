import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    return json.loads((ROOT / "evals" / name).read_text())


def test_classification_eval_exists_and_sane():
    d = _load("classification.json")
    assert d["n_test"] == 1000
    for name, m in d["models"].items():
        assert 0.5 <= m["roc_auc"] <= 1.0, name
        assert 0.0 <= m["accuracy"] <= 1.0
    assert "decision_tree" in d["models"]
    assert "keras_mlp" in d["models"]


def test_regression_eval_sane():
    d = _load("regression.json")
    assert "ridge" in d["models"]
    assert "random_forest_regressor" in d["models"]
    assert "keras_mlp" in d["models"]
    for m in d["models"].values():
        assert m["rmse"] > 0
        assert -1.0 <= m["r2"] <= 1.0


def test_clustering_eval_sane():
    d = _load("clustering.json")
    assert -1.0 <= d["silhouette_score"] <= 1.0
    assert sum(d["cluster_sizes"]) == 5000
    assert len(d["profile"]) == 4


def test_ablation_eval_present():
    d = _load("ablation.json")
    assert "clinical_only" in d
    assert "clinical_plus_genomic" in d
    assert "genomic_only" in d


def test_pharmacogenomics_eval_present():
    d = _load("pharmacogenomics.json")
    assert "with_markers" in d
    assert "without_markers" in d
    assert "roc_auc_gain_with_markers" in d
