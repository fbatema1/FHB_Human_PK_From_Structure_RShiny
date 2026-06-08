#!/bin/bash
#SBATCH --job-name=pk_xgb
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/xgb_%j.out
#SBATCH --error=logs/xgb_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=YOUR_EMAIL@unc.edu

module load anaconda/2023.03
conda activate pkip-env

cd /path/to/pk-predictor   # UPDATE THIS

mkdir -p logs models/saved/xgb

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"

python training/train_xgb.py

echo "Job finished: $(date)"
