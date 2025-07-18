# Computer‑Assisted Drug Design for EGFR‑Driven Lung Cancer

> **A self‑directed, end‑to‑end exploration of modern cheminformatics and machine‑learning techniques applied to early‑stage oncology drug discovery.**

---

## 1. Project Rationale

Small‑cell lung cancer (SCLC) remains one of the most aggressive solid tumours, with limited targeted‑therapy options and a five‑year survival rate below 10 %. Although epidermal growth‑factor receptor (EGFR) inhibitors are geared toward non‑small‑cell lung cancer (NSCLC), EGFR signalling is increasingly recognised as a vulnerability in select SCLC subtypes.  This repository documents a **workflow** that mines public chemical/bioactivity data to surface novel EGFR‑modulating chemotypes while proactively assessing developability and safety liabilities.

---
## 2. End‑to‑End Workflow

-  Ligand‑Based Virtual Screening (LBVS)

Library preparation – Clean and standardise the ChEMBL compunds known active on EGFR to obtain a modelling‑ready compound set.

Fingerprint generation – Compute three complementary fingerprints—ECFP4, ECFP6, and MACCS‑166—for every molecule.

Similarity search – Perform Tanimoto‑based searches against a curated set of compounds similar to tyrosine‑kinase inhibitors (TKIs).

Consensus ranking – Fuse the three ranked lists to yield a single hit list enriched for TKIs-likely.

-  Early‑Stage Filtering

Drug‑likeness rules – Apply Lipinski rules of five to remove poor oral candidates.

Liability alerts – Screen out PAINS scaffolds to avoid promiscuous or assay‑interfering compounds.

-  hERG Liability QSAR

Model panel – Train SGD, Logistic Regression, SVM, and Random Forest classifiers on the Therapeutics Data Commons hERG blocker dataset.

Model selection – Pick the best algorithm (Random Forest) using ROC‑AUC, PR‑AUC, and MCC.

Risk scoring – Predict hERG‑blocker probability for virtual‑screening hits and for FDA‑approved TKIs (benchmark).

- CYP450 Multi‑Isozyme ADMET

Data featurisation – Extract molecular features for five TDC CYP inhibition datasets (CYP2C19, CYP2D6, CYP3A4, CYP1A2, CYP2C9).

Learning strategy – Fit both one‑vs‑rest and classifier‑chain Random Forest models to capture multilabel patterns.

Evaluation – Report per‑isozyme metrics and global multilabel scores (Hamming loss, subset accuracy, macro ROC‑AUC).

-  Solubility Regression

Dataset – AqSolDB accessed via the TDC API.

Algorithms – Linear Regression, Support‑Vector Regression (SVR), and RANSAC.

Diagnostics – Generate bias‑variance and learning curves; assess applicability domain using bounding‑box and convex‑hull methods.

-  (Planned) Structure‑Based and Generative Extensions

Docking workflows – Deploy structure‑based virtual screening against EGFR crystal structures.

De novo design – Explore variational auto‑encoders (VAEs) and graph neural‑network (GNN) reinforcement learning for scaffold generation.


## 3. Data Sources

* **ChEMBL** — bioactivity & structures (2024‑12).^3
* **Therapeutics Data Commons (TDC)** — standardized ADMET sets.^4
* **AqSolDB** — curated aqueous solubility measurements.^5
* **PDB 4HJO** — EGFR tyrosine‑kinase domain co‑crystal.^6


---


## 6. Expected Outcomes

* **Ranked hit list** of EGFR‑like chemotypes free from obvious liability flags.
* **ADMET risk dashboard** (hERG, CYP‑inhibition, solubility) to triage hits before wet‑lab validation.

---
