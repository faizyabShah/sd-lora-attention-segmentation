#!/bin/bash
#SBATCH --job-name=merge-ca5-selca-allsa-coco
#SBATCH --output=logs/merge_ca5_selca_allsa_coco_%j.out
#SBATCH --error=logs/merge_ca5_selca_allsa_coco_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation

python scripts/merge_revised_candidate.py \
    --dataset coco_cap \
    --output-root outputs/segmentation/ca5_30_selected_ca_all_sa_heads \
    --method-key ca_steps_05_30_selected_ca_all_sa_heads \
    --method-label "CA calls 5--30, selected CA heads, and all SA heads"
