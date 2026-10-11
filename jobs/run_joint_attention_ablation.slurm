#!/bin/bash
#SBATCH --job-name=joint-attn-ablation
#SBATCH --output=logs/joint_attention_ablation_%j.out
#SBATCH --error=logs/joint_attention_ablation_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail

cd /home/shafa/sd-lora-attention-segmentation

python scripts/run_joint_attention_ablation.py \
    --capture-dir outputs/attention/voc_sim \
    --output-dir outputs/segmentation/voc_sim_100_joint_time_resolution \
    --threshold 0.4 \
    --normalization minmax \
    --branch 1
