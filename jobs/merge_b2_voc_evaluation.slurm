#!/bin/bash
#SBATCH --job-name=b2_voc_merge
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/b2_voc_merge_%j.log
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
export PYTHONDONTWRITEBYTECODE=1
cd /home/shafa/sd-lora-attention-segmentation
mkdir -p logs
python scripts/finalize_b2_token_package.py

