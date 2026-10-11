#!/bin/bash
#SBATCH --job-name=voc-attn-smoke
#SBATCH --output=logs/voc_attn_smoke_%j.out
#SBATCH --error=logs/voc_attn_smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation

python scripts/run_attention_analysis.py \
    --class-shard 0 \
    --max-samples 1 \
    --panels-per-class 1 \
    --device cuda \
    --storage-dtype float16 \
    --model-dtype float32 \
    --output-dir outputs/analysis/voc_sim_attention_report_smoke
