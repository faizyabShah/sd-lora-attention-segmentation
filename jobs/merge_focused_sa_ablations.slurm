#!/bin/bash
#SBATCH --job-name=voc-focused-merge
#SBATCH --output=logs/voc_focused_merge_%j.out
#SBATCH --error=logs/voc_focused_merge_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation

python scripts/merge_focused_sa_ablations.py \
    --input-dir outputs/segmentation/voc_sim_full_focused_sa_p1_p4 \
    --expected-shards 20
