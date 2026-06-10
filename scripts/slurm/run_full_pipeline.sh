#!/bin/bash
#SBATCH --job-name=pk_full_pipeline
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_%j.out
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fbatema1@unc.edu

# =============================================================================
# Full Wadhams PK Predictor pipeline — scaffold split edition
#
# Steps:
#   1. Scaffold split  (already done — skipped if outputs exist)
#   2. Featurization
#   3. Train RF
#   4. Train XGB
#   5. Train GNN
#   6. Conformal calibration
#
# After this finishes, upload models to HuggingFace:
#   $PYTHON api/upload_models.py
# =============================================================================

set -e

# ── Use pkip-env Python directly (bypasses conda activate in batch jobs) ──────
PYTHON=/nas/longleaf/home/fbatema1/.conda/envs/pkip-env/bin/python

cd /nas/longleaf/home/fbatema1/pk-predictor

mkdir -p logs models/saved/scaffold_rf models/saved/scaffold_xgb models/saved/scaffold_gnn models/saved/scaffold_conformal

echo "============================================="
echo "Wadhams PK Predictor — Full Pipeline"
echo "Started:  $(date)"
echo "Node:     $(hostname)"
echo "CPUs:     $SLURM_CPUS_PER_TASK"
echo "Python:   $($PYTHON --version)"
echo "============================================="

# ── Step 1: Scaffold split ────────────────────────────────────────────────────
echo ""
echo "[1/6] Scaffold split..."
$PYTHON scripts/data_curation/scaffold_split.py

# ── Step 2: Featurization ─────────────────────────────────────────────────────
echo ""
echo "[2/6] Featurization (scaffold train + test)..."
$PYTHON features/run_featurization_scaffold.py

# ── Step 3: Train RF ──────────────────────────────────────────────────────────
echo ""
echo "[3/6] Training Random Forest (CL)..."
$PYTHON training/train_rf_scaffold.py

# ── Step 4: Train XGB ─────────────────────────────────────────────────────────
echo ""
echo "[4/6] Training XGBoost (Vd)..."
$PYTHON training/train_xgb_scaffold.py

# ── Step 5: Train GNN ─────────────────────────────────────────────────────────
echo ""
echo "[5/6] Training GNN..."
$PYTHON training/train_gnn_scaffold.py

# ── Step 6: Conformal calibration ─────────────────────────────────────────────
echo ""
echo "[6/6] Conformal calibration..."
$PYTHON training/calibrate_conformal_scaffold.py

echo ""
echo "============================================="
echo "Pipeline complete: $(date)"
echo "============================================="
echo ""
echo "Next step — upload models to HuggingFace:"
echo "  $PYTHON api/upload_models.py"
