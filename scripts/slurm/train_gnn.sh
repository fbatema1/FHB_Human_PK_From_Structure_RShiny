#!/bin/bash
#SBATCH --job-name=pk_gnn
#SBATCH --partition=gpu           # GPU partition on Longleaf
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1              # request 1 GPU
#SBATCH --time=16:00:00
#SBATCH --output=logs/gnn_%j.out
#SBATCH --error=logs/gnn_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@unc.edu

# ── Environment ───────────────────────────────────────────────────────────────
module load anaconda/2023.03
module load cuda/12.1         # adjust to match your Longleaf CUDA version
conda activate pkip-env

# ── Navigate to project root ──────────────────────────────────────────────────
cd /path/to/pk-predictor   # UPDATE THIS to your Longleaf project path

# ── Create output dirs if needed ─────────────────────────────────────────────
mkdir -p logs models/saved/gnn

# ── Run ───────────────────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPU: $CUDA_VISIBLE_DEVICES"

python training/train_gnn.py

echo "Job finished: $(date)"
