#!/bin/bash
#SBATCH --job-name=smoke-voc-clean-ca
#SBATCH --output=logs/smoke_voc_clean_ca_%j.out
#SBATCH --error=logs/smoke_voc_clean_ca_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation
mkdir -p logs

python scripts/run_clean_ca_ablations.py \
    --class-shard 0 \
    --max-samples 1 \
    --output-dir outputs/segmentation/voc_sim_clean_ca_ablations_smoke \
    --device cuda \
    --storage-dtype float16 \
    --model-dtype float32 \
    --threshold 0.45 \
    --normalization minmax \
    --exhaustive-batch-size 256
