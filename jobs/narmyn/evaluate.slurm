#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/narmyn_eval_%x_%j.log
#
# All flags pass through to 07_evaluate.py.
#
#   # pick the best checkpoint (fid/kid only -- fast, 50 cells)
#   sbatch --job-name=eval_sweep jobs/narmyn/evaluate.slurm --sweep --metrics fid,kid
#
#   # full metric suite on the chosen configurations
#   sbatch --job-name=eval_final jobs/narmyn/evaluate.slurm \
#          --sets frozen rank8 rank16 rank32 rank64 rank128
#
# CONDA_ENV=gen lets this run on erdos/neumann (their drivers cannot load
# trdp's cu130 torch), provided the eval deps are installed there too.

set -eo pipefail

CONDA_ENV="${CONDA_ENV:-trdp}"

source ~/.bashrc
conda activate "${CONDA_ENV}"

REPO="${REPO:-$HOME/sd-lora-attention-segmentation}"
# Data/outputs tree (cropped images, captions, weights, generations).
export TRDP_ROOT="${TRDP_ROOT:-$HOME/TRDP}"
cd "$REPO"
mkdir -p logs "$TRDP_ROOT/output/logs" "$TRDP_ROOT/output/evaluation"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "=== env=${CONDA_ENV} node=$(hostname -s) job=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

python scripts/narmyn/07_evaluate.py "$@"
