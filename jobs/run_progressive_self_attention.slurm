#!/bin/bash
#SBATCH --job-name=sa-progressive
#SBATCH --output=logs/self_attention_progressive_%j.out
#SBATCH --error=logs/self_attention_progressive_%j.err
#SBATCH --partition=gpu_lowpriority
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation

: "${SEQUENCE:?Submit with --export=ALL,SEQUENCE=8-16-...}"
SEQUENCE_NAME=${SEQUENCE//,/-}

python scripts/run_progressive_self_attention.py \
    --sequence "$SEQUENCE" \
    --capture-dir outputs/attention/voc_sim \
    --output-dir "outputs/segmentation/voc_sim_100_self_attention/progressive_${SEQUENCE_NAME}" \
    --device cuda
