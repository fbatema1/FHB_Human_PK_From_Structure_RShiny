#!/bin/bash
#SBATCH --job-name=pk_v2_pipeline
#SBATCH --partition=volta-gpu
#SBATCH --qos=gpu_access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:tesla_v100-sxm2-16gb:1
#SBATCH --time=24:00:00
#SBATCH --output=/nas/longleaf/home/fbatema1/pk-predictor/logs/v2_pipeline_%j.out
#SBATCH --error=/nas/longleaf/home/fbatema1/pk-predictor/logs/v2_pipeline_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fbatema1@unc.edu

PYTHON=/nas/longleaf/home/fbatema1/.conda/envs/pkip-env/bin/python
cd /nas/longleaf/home/fbatema1/pk-predictor

mkdir -p logs models/saved/v2_rf models/saved/v2_xgb models/saved/v2_gnn models/saved/v2_conformal

echo "============================================="
echo "Wadhams V2 Pipeline"
echo "Started: $(date)"
echo "Node:    $(hostname)"
echo "============================================="

echo "[1/4] Training RF (v2)..."
$PYTHON v2/scripts/train_rf_v2.py

echo "[2/4] Training XGB (v2)..."
$PYTHON v2/scripts/train_xgb_v2.py

echo "[3/4] Training GNN (v2)..."
$PYTHON v2/scripts/train_gnn_v2.py

echo "[4/4] Conformal calibration (v2)..."
$PYTHON v2/scripts/calibrate_conformal_v2.py

echo "============================================="
echo "V2 Pipeline complete: $(date)"
echo "============================================="
