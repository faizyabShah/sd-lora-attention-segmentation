#!/bin/bash
#SBATCH --job-name=voc-stream-merge
#SBATCH --output=logs/voc_stream_merge_%j.out
#SBATCH --error=logs/voc_stream_merge_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation

python scripts/merge_streaming_voc_ablations.py \
    --input-dir outputs/segmentation/voc_sim_full_streaming \
    --expected-shards 20

