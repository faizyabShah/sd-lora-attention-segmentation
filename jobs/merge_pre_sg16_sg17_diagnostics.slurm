#!/bin/bash
#SBATCH --job-name=pre-sg16-sg17-merge
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4
python scripts/finalize_pre_sg16_sg17_diagnostics.py
