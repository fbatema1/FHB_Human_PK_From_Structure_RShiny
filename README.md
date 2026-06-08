# PK Predictor

**Predicting human pharmacokinetic parameters from chemical structure (SMILES)**

A machine learning pipeline to predict human clearance (CL), volume of distribution (Vd), 
half-life (t½), and terminal elimination rate constant (λz) from SMILES strings, with 
calibrated 95% confidence intervals via split conformal prediction.

---

## Parameters Predicted

| Parameter | Units | Method |
|-----------|-------|--------|
| CL | mL/min/kg | Direct prediction |
| Vd (VDss) | L/kg | Direct prediction |
| t½ | h | Derived: 0.693 × Vd / CL |
| λz | 1/h | Direct prediction |

---

## Models

Random Forest, XGBoost, and Graph Neural Network (AttentiveFP via PyTorch Geometric) 
are trained and tuned independently for each parameter. The best-performing model per 
parameter is selected based on GMFE and R² on a held-out test set, then assembled into 
a hybrid predictor.

**Performance targets:** GMFE < 1.5, R² > 0.7 for each parameter.

---

## Repository Structure

```
pk-predictor/
├── data/
│   ├── raw/                  # Source datasets (Lombardo, ChEMBL, Enamine)
│   └── processed/            # Featurized outputs (generated, not committed)
├── docs/
│   ├── decisions/            # Architectural Decision Records (ADRs)
│   ├── results/figures/      # Plots and performance summaries
│   └── references/           # Source literature
├── notebooks/                # Exploratory analysis and results
├── scripts/
│   ├── data_curation/        # Dataset assembly and ChEMBL lookup scripts
│   └── slurm/                # UNC Longleaf cluster submission scripts
├── features/                 # RDKit descriptor and PyG graph featurization
├── models/                   # RF, XGBoost, GNN model definitions
├── training/                 # Training loops, Optuna tuning, conformal calibration
├── evaluation/               # Metrics (GMFE, R², RMSE)
├── hybrid/                   # Per-parameter model routing and ensemble
├── api/                      # FastAPI prediction endpoint
├── shiny/                    # R Shiny front-end
└── tests/                    # Unit tests
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate pk-predictor
```

---

## Usage

*(To be completed)*

---

## Citation

*(To be completed upon publication)*

---

## License

*(To be decided — MIT or Apache 2.0)*
