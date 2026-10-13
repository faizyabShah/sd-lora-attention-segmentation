#!/bin/bash
#SBATCH --job-name=merge-voc-clean-ca
#SBATCH --output=logs/merge_voc_clean_ca_%j.out
#SBATCH --error=logs/merge_voc_clean_ca_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation

python scripts/merge_clean_ca_ablations.py
