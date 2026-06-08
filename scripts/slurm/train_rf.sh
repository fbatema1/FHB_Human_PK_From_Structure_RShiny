#!/bin/bash
#SBATCH --job-name=pk_rf
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rf_%j.out
#SBATCH --error=logs/rf_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@unc.edu

# ── Environment ───────────────────────────────────────────────────────────────
module load anaconda/2023.03
conda activate pkip-env

# ── Navigate to project root ──────────────────────────────────────────────────
cd /path/to/pk-predictor   # UPDATE THIS to your Longleaf project path

# ── Create logs dir if needed ─────────────────────────────────────────────────
mkdir -p logs models/saved/rf

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

python training/train_rf.py

echo "Job finished: $(date)"
