#!/bin/bash
#SBATCH --job-name=ovam-attn-100
#SBATCH --output=logs/ovam_attn_100_%j.out
#SBATCH --error=logs/ovam_attn_100_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail

cd /home/shafa/sd-lora-attention-segmentation

python scripts/capture_ovam_attention.py \
    --count 100 \
    --start-index 0 \
    --storage-dtype float16 \
    --min-free-gb 50 \
    --output-dir outputs/attention/voc_sim
