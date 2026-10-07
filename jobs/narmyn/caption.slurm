#!/bin/bash
#SBATCH --job-name=caption
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/narmyn_caption_%j.log
# renyi carries 2x a100, which beats neumann's single v100 for fp16 inference.
# We ask for 1 GPU because the script is single-device, leaving renyi's other
# a100 free. To use neumann instead, swap the two lines above for:
#   #SBATCH --nodelist=neumann
#   #SBATCH --gres=gpu:v100:1
# stderr is merged into --output. Submit from ~/TRDP so the log path resolves.

set -eo pipefail

source ~/.bashrc
# Not trdp: Florence-2's remote code is written against transformers 4.x and
# dies on trdp's 5.6.2 ('Florence2LanguageConfig' has no attribute
# 'forced_bos_token_id'). `caption` is a clone of trdp pinned to
# transformers==4.44.2, so trdp stays intact for LoRA training.
conda activate caption

REPO="${REPO:-$HOME/sd-lora-attention-segmentation}"
# Data/outputs tree (cropped images, captions, weights, generations).
export TRDP_ROOT="${TRDP_ROOT:-$HOME/TRDP}"
cd "$REPO"
mkdir -p logs "$TRDP_ROOT/output/logs" output/captions

# Keep the Florence-2 download in the shared HF cache rather than re-fetching.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

python scripts/narmyn/02_caption.py "$@"
