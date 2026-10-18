#!/bin/bash
#SBATCH --job-name=b2rep_merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-gen}"
export PYTHONDONTWRITEBYTECODE=1
python scripts/finalize_b2_token_package.py --root outputs/segmentation/controlled_attention_cv/token_learning_b2_replication
