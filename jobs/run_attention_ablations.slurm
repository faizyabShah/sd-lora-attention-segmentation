#!/bin/bash
#SBATCH --job-name=attention-ablations
#SBATCH --output=logs/attention_ablations_%j.out
#SBATCH --error=logs/attention_ablations_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail

cd /home/shafa/sd-lora-attention-segmentation

python scripts/run_attention_ablations.py \
    --ablations time_window resolution \
    --capture-dir outputs/attention/voc_sim \
    --output-dir outputs/segmentation/voc_sim_100_ablations \
    --threshold 0.4 \
    --normalization minmax \
    --branch 1
