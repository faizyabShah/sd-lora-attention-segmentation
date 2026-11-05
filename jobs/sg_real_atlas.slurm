#!/bin/bash
# SG-real-atlas capture (protocol Section 137). Any GPU node except wald.
#SBATCH --job-name=sg-real-atlas
#SBATCH --partition=gpu
#SBATCH --exclude=wald
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
cd /home/shafa/sd-lora-attention-segmentation
source ~/.bashrc
conda activate trdp
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/run_sg_real_atlas.py "$@"
