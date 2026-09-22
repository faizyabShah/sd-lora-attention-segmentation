#!/bin/bash
#SBATCH --job-name=cityscapes-caption
#SBATCH --output=caption_%j.out
#SBATCH --error=caption_%j.err

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=2:00:00

# =========================
# SETUP
# =========================

source ~/.bashrc
conda activate trdp   # change if your env name is different

echo "JOB ID: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# =========================
# OPTIONAL: SPEED OPTIMIZATIONS
# =========================

export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME

# avoid tokenizer warnings
export TOKENIZERS_PARALLELISM=false

# =========================
# RUN SCRIPT
# =========================

python crop_and_caption.py \
    --src ./leftImg8bit \
    --dst ./cityscapes_cropped \
    --save_every 100

echo "=== DONE ==="