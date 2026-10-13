#!/bin/bash
#SBATCH --job-name=merge-ca10-allsa
#SBATCH --output=logs/merge_ca10_allsa_%j.out
#SBATCH --error=logs/merge_ca10_allsa_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation

python scripts/merge_revised_candidate.py
