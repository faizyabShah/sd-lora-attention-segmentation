#!/bin/bash
#SBATCH --job-name=ca-layer-ablation
#SBATCH --output=logs/cross_attention_layer_ablation_%j.out
#SBATCH --error=logs/cross_attention_layer_ablation_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail

cd /home/shafa/sd-lora-attention-segmentation

python scripts/run_cross_attention_layer_ablation.py \
    --capture-dir outputs/attention/voc_sim \
    --output-dir outputs/segmentation/voc_sim_100_cross_attention_layers \
    --threshold 0.4 \
    --normalization minmax \
    --branch 1
