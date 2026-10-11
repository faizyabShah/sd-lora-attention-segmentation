#!/bin/bash
#SBATCH --job-name=sa-screen
#SBATCH --output=logs/self_attention_screen_%j.out
#SBATCH --error=logs/self_attention_screen_%j.err
#SBATCH --partition=gpu_lowpriority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation

: "${RESOLUTION:?Submit with --export=ALL,RESOLUTION=8|16|32|64}"

python scripts/run_self_attention_screen.py \
    --resolution "$RESOLUTION" \
    --capture-dir outputs/attention/voc_sim \
    --output-dir "outputs/segmentation/voc_sim_100_self_attention/r${RESOLUTION}" \
    --device cuda
