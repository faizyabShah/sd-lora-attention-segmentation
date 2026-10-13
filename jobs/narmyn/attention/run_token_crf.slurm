#!/bin/bash
#SBATCH --job-name=token_crf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/token_crf_%A_%a.log
#
# OVAM optimized tokens + DenseCRF on the earlier final method (exploratory).
# One array task per class. Pass the dataset and any extra flags through:
#
#   sbatch jobs/narmyn/attention/run_token_crf.slurm --dataset voc
#   sbatch jobs/narmyn/attention/run_token_crf.slurm --dataset coco
#
# Set CONDA_ENV to an environment whose torch build runs on the node and which
# has SimpleCRF (or pass --crf-site <dir containing denseCRF>). Run from the
# repository root.
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p logs
echo "node=$(hostname -s) env=${CONDA_ENV:-trdp} shard=${SLURM_ARRAY_TASK_ID}"
python scripts/narmyn/attention/run_token_crf.py --class-shard "${SLURM_ARRAY_TASK_ID}" "$@"
