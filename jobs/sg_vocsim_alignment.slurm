#!/bin/bash
# VOC-Sim replication of the SG-real-atlas alignment diagnostics (stats only). Any GPU node except wald.
#SBATCH --job-name=sg-vocsim-align
#SBATCH --partition=gpu
#SBATCH --exclude=wald
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
cd /home/shafa/sd-lora-attention-segmentation
source ~/.bashrc
conda activate trdp
set -u
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/run_sg_vocsim_alignment.py "$@"
