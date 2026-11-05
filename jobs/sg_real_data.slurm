#!/bin/bash
# SG-real-data: frozen BRACE on real photos. V100 nodes only (erdos, neumann).
#SBATCH --job-name=sg-real-data
#SBATCH --partition=gpu
#SBATCH --exclude=renyi,wald
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
cd /home/shafa/sd-lora-attention-segmentation
source ~/.bashrc
conda activate trdp
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
python scripts/run_sg_real_data.py "$@"
