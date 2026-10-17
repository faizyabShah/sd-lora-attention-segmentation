#!/bin/bash
#SBATCH --job-name=b2a_tokens
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-58
#SBATCH --output=logs/b2a_tokens_%A_%a.log
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
cd /home/shafa/sd-lora-attention-segmentation
mkdir -p logs
python scripts/optimize_b2_tokens.py --track b2a --task-id "${SLURM_ARRAY_TASK_ID}"

