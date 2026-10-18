#!/bin/bash
#SBATCH --job-name=sg5_final
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#
# Fold selection, AN1 diagnostics and COCO-Cap transfer for SG5 (protocol Section 66).
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-gen}"
export PYTHONDONTWRITEBYTECODE=1
python scripts/narmyn/attention/finalize_sg5.py
