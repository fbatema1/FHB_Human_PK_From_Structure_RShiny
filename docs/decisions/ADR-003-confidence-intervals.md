# ADR-003: Confidence Interval Method

**Date:** 2026-06-07  
**Status:** Decided

## Context
The tool must report 95% confidence intervals alongside point predictions. 
The CI method must work consistently across RF, XGB, and GNN (different model families).

## Decision
**Split conformal prediction** — a model-agnostic, distribution-free framework 
that provides valid marginal coverage guarantees.

Procedure:
1. Reserve a calibration set (~200 compounds from training split)
2. Compute nonconformity scores (|y_true - y_pred|) on calibration set
3. Set threshold q at the (1-α)(1 + 1/n) quantile of calibration scores
4. Prediction interval: [ŷ - q, ŷ + q] on log scale, back-transformed

## Alternatives Considered
- **Bootstrap CIs:** Train N=100+ models on bootstrap samples, use prediction 
  variance. Rejected — prohibitively expensive for GNN; produces inconsistent 
  interval widths across model types.
- **Quantile regression:** Train separate upper/lower quantile models. Rejected — 
  requires additional models per parameter; harder to maintain consistency across 
  RF/XGB/GNN.
- **Bayesian methods (MC Dropout, deep ensembles):** Interesting but adds significant 
  complexity; GNN ensemble training would be very expensive on this timeline.

## Rationale
Conformal prediction gives a provable coverage guarantee (empirical coverage ≥ 95% 
on exchangeable data) regardless of model type or data distribution. Identical 
procedure for RF, XGB, and GNN — one implementation, applied uniformly.

## Consequences
- Calibration set must be held out from Optuna tuning (no data leakage)
- Coverage must be validated empirically on the test set and reported
- Intervals are symmetric on log scale; may be asymmetric on original scale 
  after back-transformation (which is scientifically appropriate for PK data)
