#!/bin/bash
#SBATCH --job-name=pre-sg19-merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# Reuses completed candidate-pixel caches; no image generation or GPU.
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
python scripts/finalize_pre_sg19_late_support.py
