#!/bin/bash
#SBATCH --job-name=sg24-merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repository root}"
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
: "${EXPERIMENT:?set EXPERIMENT=a or b}"
python scripts/finalize_sg24.py --experiment "$EXPERIMENT"
