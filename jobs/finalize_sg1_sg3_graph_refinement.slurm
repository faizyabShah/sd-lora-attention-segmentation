#!/bin/bash
#SBATCH --job-name=final-sg1-sg3
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
set -eo pipefail
repo="${SLURM_SUBMIT_DIR:?submit from the repository root}"
cd "$repo"
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
set -u
export PYTHONDONTWRITEBYTECODE=1
python scripts/finalize_sg1_sg3_graph_refinement.py
