#!/bin/bash
#SBATCH --job-name=paper_figures
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/nas/longleaf/home/fbatema1/pk-predictor/logs/figures_%j.out
#SBATCH --error=/nas/longleaf/home/fbatema1/pk-predictor/logs/figures_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=fbatema1@unc.edu

PYTHON=/nas/longleaf/home/fbatema1/.conda/envs/pkip-env/bin/python
cd /nas/longleaf/home/fbatema1/pk-predictor

$PYTHON scripts/generate_paper_figures.py
