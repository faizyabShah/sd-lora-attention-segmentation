#!/bin/bash
#SBATCH --job-name=b2_voc_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/b2_voc_eval_%A_%a.log
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
cd /home/shafa/sd-lora-attention-segmentation
mkdir -p logs
python scripts/run_b2_token_evaluation.py --dataset voc --class-shard "${SLURM_ARRAY_TASK_ID}" \
  --crf-site "${CRF_SITE:-$HOME/TRDP/output/ovam_token_pilot/crf_site}"

