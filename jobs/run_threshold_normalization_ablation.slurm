#!/bin/bash
#SBATCH --job-name=threshold-normalization
#SBATCH --output=logs/threshold_normalization_%j.out
#SBATCH --error=logs/threshold_normalization_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail

cd /home/shafa/sd-lora-attention-segmentation

python scripts/run_threshold_normalization_ablation.py \
    --capture-dir outputs/attention/voc_sim \
    --output-dir outputs/segmentation/voc_sim_100_threshold_normalization \
    --branch 1 \
    --working-size 64
