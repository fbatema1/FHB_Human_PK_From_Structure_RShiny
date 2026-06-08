# ADR-005: Hyperparameter Optimization Strategy

**Date:** 2026-06-07  
**Status:** Decided

## Context
Each model × parameter combination requires hyperparameter tuning. 
The combined feature + hyperparameter search space is large.

## Decision
- **Framework:** Optuna with TPE (Tree-structured Parzen Estimator) sampler
- **Trials:** RF: 300, XGB: 300, GNN: 150–200 per parameter
- **Pruning:** MedianPruner (kills underperforming trials early)
- **Parallelism:** Shared PostgreSQL/SQLite study DB on UNC Longleaf; 
  multiple SLURM jobs contribute to the same study simultaneously
- **Objective:** Minimize GMFE on 5-fold CV validation set (log scale RMSE as proxy)
- **Feature selection:** `top_n_features` (int, 20–140) tuned as a hyperparameter 
  over SHAP-ranked descriptor list

## Alternatives Considered
- **Grid/random search:** Rejected — inefficient for high-dimensional spaces
- **100 trials:** Considered but insufficient given combined feature + HP search space
- **500+ GNN trials:** Rejected — each GNN trial involves full model training; 
  150-200 with pruning is equivalent to many more effectively evaluated configurations

## Rationale
TPE is sample-efficient for mixed continuous/integer/categorical search spaces. 
Pruning reduces wall-clock time by 40–60% for XGB and GNN. Cluster parallelism 
means 300 trials can complete in hours rather than days.

## Consequences
- Total: ~3,100 Optuna trials across all 12 studies
- SLURM array jobs needed for parallel execution
- Study databases must be backed up (results are irreplaceable)
- GNN pruning requires epoch-level reporting to Optuna (intermediate values)
