# ADR-002: Feature Engineering Strategy

**Date:** 2026-06-07  
**Status:** Decided

## Context
SMILES strings must be converted to numerical representations for RF and XGB. 
GNN operates on molecular graphs directly. Feature selection is important given 
collinearity among RDKit descriptors.

## Decision
**For RF and XGB:**
1. Compute all ~200 RDKit 2D descriptors + 2048-bit Morgan fingerprints (radius=2)
2. Pre-filter: remove near-zero variance descriptors and highly correlated pairs (Pearson r > 0.95)
3. Train baseline RF/XGB and compute SHAP values per parameter
4. Rank features by mean |SHAP| per parameter independently
5. Optuna tunes `top_n_features` (integer, 20–140) alongside model hyperparameters

**For GNN (AttentiveFP):**
- Atom features: atomic number, formal charge, aromaticity, hybridization, H count, ring membership
- Bond features: bond type, conjugation, ring membership, stereo
- No descriptor pre-processing needed

## Alternatives Considered
- **Optuna selecting individual features (binary on/off):** Rejected — 2^200 search 
  space is intractable. Optuna tuning a threshold over a SHAP-ranked list is equivalent 
  but tractable.
- **PCA dimensionality reduction:** Rejected — loses interpretability, which matters 
  for a published tool.
- **Fingerprints only (no descriptors):** Rejected — descriptors capture physicochemical 
  properties (logP, TPSA, MW) that are mechanistically relevant to CL and Vd.

## Rationale
SHAP values provide more reliable feature importance than built-in RF/XGB importance 
(which is biased toward high-cardinality features). Tuning `top_n_features` via Optuna 
keeps feature selection jointly optimized with hyperparameters without combinatorial 
explosion.

## Consequences
- Feature importance is parameter-specific: CL, Vd, λz will select different descriptor subsets
- SHAP computation adds ~10-30 min to baseline training step (one-time cost)
- Final feature sets must be saved and versioned for reproducibility
