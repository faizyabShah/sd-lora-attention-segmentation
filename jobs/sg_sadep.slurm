#!/bin/bash
# Dense CLIP (SCLIP port) faith check and capture (GPU). Any GPU node except wald.
#SBATCH --job-name=sg-sadep
#SBATCH --partition=gpu
#SBATCH --exclude=wald
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
cd /home/shafa/sd-lora-attention-segmentation
source ~/.bashrc
conda activate trdp
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/sg_sa_prompt_dep.py
