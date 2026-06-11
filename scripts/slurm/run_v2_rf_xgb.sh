#!/bin/bash
#SBATCH --job-name=pk_v2_rfxgb
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/nas/longleaf/home/fbatema1/pk-predictor/logs/v2_rfxgb_%j.out
#SBATCH --error=/nas/longleaf/home/fbatema1/pk-predictor/logs/v2_rfxgb_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fbatema1@unc.edu

# =============================================================================
# Wadhams v2 — Track A (physchem-enriched), CPU-only
# Trains RF (CL) and XGB (Vd) on the 2210-feature v2 matrix and evaluates on
# the v2 scaffold test set. No GPU, no GNN (GNN doesn't use tabular features
# yet), no conformal (deferred until we confirm the models improve).
# =============================================================================

set -e

PYTHON=/nas/longleaf/home/fbatema1/.conda/envs/pkip-env/bin/python
cd /nas/longleaf/home/fbatema1/pk-predictor
mkdir -p logs models/saved/v2_rf models/saved/v2_xgb

echo "============================================="
echo "Wadhams v2 Track A (RF + XGB)"
echo "Started:  $(date)"
echo "Node:     $(hostname)"
echo "Python:   $($PYTHON --version)"
echo "============================================="

echo ""
echo "[1/2] Training Random Forest (CL + Vd)..."
$PYTHON v2/scripts/train_rf_v2.py

echo ""
echo "[2/2] Training XGBoost (CL + Vd)..."
$PYTHON v2/scripts/train_xgb_v2.py

echo ""
echo "============================================="
echo "v2 Track A complete: $(date)"
echo "Results:"
echo "  models/saved/v2_rf/rf_results.json"
echo "  models/saved/v2_xgb/xgb_results.json"
echo "Compare against v1 scaffold (RF CL 2.77 / Vd 2.19, XGB CL 2.75 / Vd 2.17)"
echo "============================================="
