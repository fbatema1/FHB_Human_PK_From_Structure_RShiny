# ADR-001: Model Selection

**Date:** 2026-06-07  
**Status:** Decided

## Context
We need to predict four human PK parameters (CL, Vd, t½, λz) from SMILES strings. 
The dataset is ~2000 compounds. We need GMFE < 1.5 and R² > 0.7 per parameter.

## Decision
Train three model families independently for each parameter:
1. **Random Forest (RF)** — scikit-learn
2. **XGBoost** — gradient boosted trees
3. **Graph Neural Network (GNN)** — AttentiveFP via PyTorch Geometric

Select the best-performing model per parameter and assemble into a hybrid predictor.

## Alternatives Considered
- **Single model for all parameters:** Rejected — different parameters likely have 
  different optimal architectures and feature sets.
- **Deep neural network on descriptors only:** Considered but GNN is preferred as it 
  operates directly on molecular graph structure, capturing topology not encoded by 
  descriptors alone.
- **Ensemble/averaging across all three models:** May be explored if single-model 
  targets are not met, but adds complexity to the CI layer.

## Rationale
- RF and XGB are strong baselines for tabular/descriptor data with this dataset size
- GNN operates on molecular graphs directly — complementary information source
- Per-parameter selection maximizes performance on each endpoint independently
- Dataset size (~2000) is workable for all three; GNN may underperform RF/XGB at 
  this scale, which is acceptable

## Consequences
- 12 Optuna studies required (3 models × 4 parameters)
- GNN requires GPU compute — UNC Longleaf cluster allocated
- Hybrid model must handle routing logic and consistent CI output across model types
