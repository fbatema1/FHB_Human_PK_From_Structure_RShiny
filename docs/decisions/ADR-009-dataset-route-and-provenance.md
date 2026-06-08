# ADR-009: Dataset Route of Administration and Provenance

**Date:** 2026-06-08  
**Status:** Decided

## Context
The master dataset combines compounds from multiple sources. Route of administration
is critical for PK data validity — CL and Vd (VDss) are only directly measurable
from intravenous (IV) dosing. Oral studies yield CL/F (apparent clearance), not
true systemic CL. We needed to confirm that all training data represents IV dosing
before proceeding with modeling.

## Decision
**All training data is confirmed IV.** No route correction is needed.

## Evidence
The original 824 compounds in the master dataset were sourced from:

> Iwata et al., *J. Chem. Inf. Model.* 2022, 62, 4057–4065
> "Predicting Total Drug Clearance and Volumes of Distribution Using the
> Machine Learning-Mediated Multimodal Method through the Imputation of
> Various Nonclinical Data"

This paper sourced its human CL and Vd data from **Lombardo et al.** (same
dataset as our Lombardo addition), which is explicitly an IV dataset. The
Iwata paper states values "were gathered during intravenous administration."

The Lombardo full dataset (1,352 compounds) added in the merge step is also
confirmed IV — it is a curated compilation of human IV pharmacokinetic studies.

Both sources trace to the same underlying IV literature data.

## Implications
- CL values represent true systemic clearance (not CL/F)
- Vd values represent true VDss at steady state
- Dataset is internally consistent with respect to route
- No bioavailability correction required

## Benchmark
The Iwata et al. paper used the same source data with XGBoost/RF and achieved:
- CL: GMFE = 1.92, within 2-fold = 66.5%
- Vd: GMFE = 1.64, within 2-fold = 71.1%

These are the benchmarks to beat. Our advantages over Iwata et al.:
- Larger dataset (~2,293 vs ~741 compounds for CL)
- Pure structure-based features (no imputed nonclinical data required)
- GNN in addition to RF/XGB
- Optuna hyperparameter tuning with pruning
- Conformal prediction CIs

## Consequences
- Route column not needed in dataset (all IV by confirmation)
- Paper methods section must cite both Lombardo et al. and Iwata et al.
  as primary data sources and acknowledge IV-only scope as a limitation
- Tool should clearly state in the Shiny UI that predictions assume
  intravenous-equivalent (systemic) PK — not oral bioavailability
- Oral PK prediction (accounting for F) is a potential future direction
