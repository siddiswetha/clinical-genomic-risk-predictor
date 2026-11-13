# Data card — synthetic only

All data in this repository is synthetic. There are no real patients,
no real genomes, no real clinical records, and no real drug data.

## Generation

- `scripts/generate_data.py` generates 5,000 patients with
  `numpy.random.default_rng(42)` — a fixed seed, so the file is
  reproducible: `python scripts/generate_data.py`.
- IDs are `SYN-00001` … `SYN-05000`. They are not linked to any person.

## Features

Clinical (11): age, sex, BMI, systolic/diastolic BP, total
cholesterol, HDL, fasting glucose, eGFR, smoking, family history.

Genomic (21): 20 synthetic SNPs, each coded 0/1/2 (number of minor
alleles), with fixed minor-allele frequencies, plus a polygenic risk
score (PRS) — a weighted sum of the SNPs, standardised. SNP names
(`snp_01` … `snp_20`) are invented; they do not correspond to real
rsIDs or real genes.

## Targets — known synthetic relationships (by construction)

- `risk_5yr` (binary): 5-year diabetes/cardiovascular risk. Its
  logit is a documented linear function of age, BMI, blood pressure,
  fasting glucose, HbA1c, smoking, family history, and PRS. It is
  a teaching signal, not a real risk model.
- `hba1c` (continuous): generated from BMI, age, fasting glucose,
  and PRS plus noise. Used as the regression target.
- `drug_responder` (binary): response to a synthetic anti-diabetic
  drug. Its logit is driven partly by `snp_03` and `snp_07`, which
  is why the pharmacogenomics ablation removes those two markers.

## What this data is NOT

- It does not reflect real biology, real disease prevalence, real
  genetics, or real drug response.
- It must not be used for clinical decisions, diagnosis, treatment,
  or patient care. See the disclaimer in `README.md`.
- Any performance number in `evals/` measures how well a model
  recovers the synthetic rules above — nothing more.
