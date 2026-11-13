# clinical-genomic-risk-predictor

> **Research / educational prototype only. Synthetic data only.
> Not medical advice. Not for clinical use. No real patients.**

A classical machine-learning project that predicts disease risk from
synthetic clinical + genomic data, with a pharmacogenomics
(drug-response) component. It was built to fill gaps in the author's
other work: healthcare / pharma / genomics / clinical / drug
discovery, regression, clustering, a standalone decision tree, and
TensorFlow — all with real, measured results on a held-out split.

## What it does

- Generates 5,000 synthetic patients (fixed seed 42): clinical
  features (age, sex, BMI, blood pressure, cholesterol, HDL,
  fasting glucose, eGFR, smoking, family history), 20 synthetic
  SNPs coded 0/1/2, and a polygenic risk score (PRS).
- **Classification** — 5-year diabetes/cardiovascular risk
  (binary): logistic regression, standalone Decision Tree, Random
  Forest, Gradient Boosting, and a TensorFlow Keras MLP.
  Compared with stratified 5-fold CV ROC-AUC, then held-out
  accuracy / precision / recall / F1 / ROC-AUC, plus threshold
  analysis and permutation importance.
- **Regression** — predict continuous HbA1c: Ridge, Random Forest
  Regressor, and a Keras MLP regressor. Reported as RMSE / MAE / R².
- **Clustering** — KMeans (k=4) patient subtypes on scaled
  clinical+genomic features, with silhouette score and a profile
  table. Clusters are exploratory groupings of synthetic data, not
  real disease subtypes.
- **Genomics ablation** — clinical-only vs clinical+genomic vs
  genomic-only (logistic regression, held-out ROC-AUC).
- **Pharmacogenomics** — drug-responder classification, with and
  without the two genomic markers (`snp_03`, `snp_07`).
- **Serving** — FastAPI with `POST /predict-risk`,
  `POST /predict-hba1c`, `GET /health`, `GET /metrics`, pydantic
  input validation, structured JSON logs, and run history saved
  to files. Dockerfile + Docker Compose included.

## Results

*All numbers below are the real measured results from
`scripts/train.py` on the held-out split (80/20, seed 42; 4,000
train / 1,000 test), saved in `evals/`. They measure recovery of
the synthetic generation rules documented in `data/DATA_CARD.md`
— nothing more. Risk prevalence in the data is 19.8%.*

### Classification — 5-year risk (held-out, n=1,000)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV ROC-AUC (5-fold) |
|---|---|---|---|---|---|---|
| Logistic regression | 0.811 | 0.574 | 0.177 | 0.270 | **0.725** | 0.738 ± 0.027 |
| Decision Tree | 0.798 | 0.466 | 0.136 | 0.211 | 0.662 | 0.665 ± 0.026 |
| Random Forest | 0.804 | 0.550 | 0.056 | 0.101 | 0.703 | 0.704 ± 0.025 |
| Gradient Boosting | 0.811 | 0.576 | 0.172 | 0.265 | 0.708 | 0.714 ± 0.025 |
| Keras MLP (TensorFlow) | 0.796 | 0.473 | 0.268 | 0.342 | 0.708 | — (held-out only) |

Recall is low at the 0.5 threshold for every model — reported,
not hidden. At threshold 0.3, the best model (logistic
regression) reaches recall 0.444 at precision 0.393. Top
permutation-importance features for logistic regression: age,
BMI, then individual SNPs.

### Regression — HbA1c (held-out)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Ridge | **0.340** | **0.275** | **0.339** |
| Random Forest Regressor | 0.346 | 0.279 | 0.316 |
| Keras MLP (TensorFlow) | 0.351 | 0.282 | 0.296 |

### Clustering — KMeans k=4

Silhouette score **0.031** — the clusters overlap heavily, as
expected for synthetic continuous data. Sizes: 1,429 / 1,190 /
1,337 / 1,044. Cluster 1 (mean PRS +1.27, mean age 55.7) has the
highest risk prevalence, 0.332; cluster 0 (mean PRS −0.38, mean
age 46.8) the lowest, 0.085. Full profile in
`evals/clustering.json`.

### Genomics ablation — logistic regression, held-out ROC-AUC

- Clinical-only: **0.690**
- Clinical + genomic: **0.725**
- Genomic-only: **0.648**
- Genomic gain: **+0.035**

### Pharmacogenomics — drug responder, held-out ROC-AUC

- With `snp_03` + `snp_07`: **0.647**
- Without them: **0.558**
- Gain: **+0.089**

Method note: the "without markers" run also excludes PRS,
because PRS is a weighted sum of the SNPs including `snp_03` /
`snp_07` — leaving it in would leak the marker signal. The
first version of this ablation left PRS in and showed a gain of
−0.0001; that version was wrong, and was corrected before
reporting.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_data.py   # writes data/patients.csv (5,000 rows)
python scripts/train.py           # trains all models, writes evals/*.json
pytest -q                         # tests
uvicorn src.clinical_genomic.app:app --host 0.0.0.0 --port 8000
# or: docker compose up --build
```

Test the API:

```bash
curl localhost:8000/health
curl -X POST localhost:8000/predict-risk -H 'Content-Type: application/json' -d \
  '{"age":55,"sex":0,"bmi":27,"sbp":130,"dbp":80,"total_cholesterol":200,"hdl":55,"fasting_glucose":95,"egfr":90,"smoking":0,"family_history":0,"prs":0.0}'
```

## Reproduce the evals

```bash
python scripts/generate_data.py
python scripts/train.py
cat evals/classification.json
cat evals/regression.json
cat evals/clustering.json
cat evals/ablation.json
cat evals/pharmacogenomics.json
```

## Project layout

```
clinical-genomic-risk-predictor/
├── data/
│   ├── DATA_CARD.md          # generation rules, what the data is NOT
│   └── patients.csv          # generated, 5,000 synthetic patients
├── src/clinical_genomic/
│   ├── data.py               # synthetic data generation
│   ├── features.py           # feature lists, preprocessing
│   ├── models.py             # Keras MLP classifier / regressor
│   └── app.py                # FastAPI service
├── scripts/
│   ├── generate_data.py
│   └── train.py              # all tasks, writes evals/*.json
├── tests/                    # pytest, 16 tests
├── evals/                    # measured results (JSON)
├── models/                   # trained models (joblib / .keras)
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml  # pip install, generate, train, pytest
```

## Honest limitations

- **Synthetic data only.** The generation rules are simple, invented
  linear/logistic functions. They do not reflect real biology,
  real disease prevalence, real genetics, or real drug response.
- **No clinical validity.** Nothing here has been validated on
  real patients, and nothing here should be used for diagnosis,
  treatment, or any clinical decision.
- **TensorFlow is CPU-only** in this repo; the MLPs are small
  (64→32 hidden units) with early stopping.
- **Serving is local / Docker only.** There is no production
  deployment, no auth, and no monitoring beyond `/metrics` and
  structured logs. The Dockerfile / Docker Compose files are
  included but were not built or tested on this VM — Docker is
  not installed here.
- HbA1c is generated from the input features by construction, so
  the regression task measures recovery of that synthetic rule.
- The Keras MLP was evaluated on the held-out split only (no
  5-fold CV); the classical models report both CV and held-out.
