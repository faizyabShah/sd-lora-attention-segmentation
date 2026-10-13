#!/bin/bash
#SBATCH --job-name=merge-onepass-allca
#SBATCH --output=logs/merge_onepass_allca_%j.out
#SBATCH --error=logs/merge_onepass_allca_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation

python scripts/merge_one_pass_no_ca_layer_filter.py
