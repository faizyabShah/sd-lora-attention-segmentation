#!/bin/bash
#SBATCH --job-name=all-layer-subsets
#SBATCH --output=logs/exhaustive_layer_search_%j.out
#SBATCH --error=logs/exhaustive_layer_search_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail

cd /home/shafa/sd-lora-attention-segmentation

python scripts/run_exhaustive_layer_search.py \
    --capture-dir outputs/attention/voc_sim \
    --output-dir outputs/segmentation/voc_sim_100_exhaustive_layer_subsets \
    --threshold 0.4 \
    --branch 1 \
    --device cuda \
    --batch-size 256
