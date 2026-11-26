#!/bin/bash
# NERVE-environment GPU job (side N's conda env, read-only use). Usage: sbatch jobs/nerve_gpu.slurm scripts/jc_swap_nerve.py <args>
#SBATCH --job-name=nerve-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
cd /home/shafa/sd-lora-attention-segmentation
export BRACE_CACHE_DIR=/home/shafa/sg_comp PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/home/sidna/miniconda3/envs/nerve/bin/python -u "$@"
