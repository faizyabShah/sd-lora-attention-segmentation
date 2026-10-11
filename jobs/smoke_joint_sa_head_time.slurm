#!/bin/bash
#SBATCH --job-name=voc-joint-smoke
#SBATCH --output=logs/voc_joint_smoke_%j.out
#SBATCH --error=logs/voc_joint_smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation

python scripts/run_joint_sa_head_time.py \
    --class-shard 0 \
    --max-samples 1 \
    --device cuda \
    --storage-dtype float16 \
    --model-dtype float32 \
    --output-dir outputs/segmentation/voc_sim_joint_sa_head_time_smoke
