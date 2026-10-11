#!/bin/bash
#SBATCH --job-name=voc-stream-smoke
#SBATCH --output=logs/voc_stream_smoke_%j.out
#SBATCH --error=logs/voc_stream_smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation

python scripts/run_streaming_voc_ablations.py \
    --class-shard 0 \
    --max-samples 2 \
    --device cuda \
    --storage-dtype float16 \
    --model-dtype float32 \
    --output-dir outputs/segmentation/streaming_smoke_fixed

