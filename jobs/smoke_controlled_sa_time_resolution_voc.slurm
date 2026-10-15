#!/bin/bash
#SBATCH --job-name=smoke-sa-tr
#SBATCH --output=logs/smoke_sa_tr_%j.out
#SBATCH --error=logs/smoke_sa_tr_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation
mkdir -p logs

python scripts/run_controlled_sa_time_resolution.py \
    --class-shard 0 \
    --max-samples 1 \
    --verify-reference \
    --output-dir outputs/segmentation/controlled_attention_cv/sa_time_resolution_smoke
