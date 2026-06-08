# ADR-008: fup Excluded as Input Feature

**Date:** 2026-06-08  
**Status:** Decided

## Context
The master dataset contains fup (fraction unbound in plasma) for 1,012 of 2,293
compounds (~44% coverage) after merging all available sources. fup is mechanistically
relevant to CL via the free drug hypothesis (CL_intrinsic relates to unbound drug),
and was initially considered as a potential input feature.

## Decision
**Exclude fup as an input feature.** It will remain in the dataset for reference
but will not be used as a predictor variable in any model.

## Rationale
- 1,233 of 2,293 compounds (~54%) are missing fup — too incomplete to use as a
  feature without imputation
- Imputation introduces additional modeling assumptions and a potential source of
  error that would be difficult to justify in a publication
- RDKit 2D descriptors (logP, TPSA, polar surface area, HBD/HBA counts) capture
  the structural determinants of plasma protein binding implicitly — the model
  can learn these relationships without explicit fup input
- Excluding fup keeps the model purely structure-based, which is scientifically
  cleaner and more useful (users won't need to know fup to get a prediction)

## Alternatives Considered
- **Use fup as a feature with median imputation:** Rejected — introduces noise
  and is hard to defend in peer review
- **Train two models (with/without fup) and compare:** Interesting scientifically
  but adds complexity and doubles the modeling work; can be explored as a future
  direction in the paper
- **Only train on the 1,012 compounds with complete fup:** Rejected — cuts the
  usable dataset by ~30%, which is too costly

## Consequences
- Input features are purely structural: RDKit 2D descriptors + Morgan fingerprints
  (RF/XGB) and molecular graph (GNN)
- fup column retained in dataset for documentation and potential future use
- Paper should note that fup was available for a subset of compounds but excluded
  for completeness reasons — cite as a limitation and future direction
