#!/bin/bash
#SBATCH --job-name=smoke-propagation
#SBATCH --output=logs/smoke_propagation_%j.out
#SBATCH --error=logs/smoke_propagation_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation
mkdir -p logs
python scripts/run_propagation_package.py --class-shard 0 --max-samples 1 \
  --verify-reference \
  --output-dir outputs/segmentation/controlled_attention_cv/propagation_package_smoke
