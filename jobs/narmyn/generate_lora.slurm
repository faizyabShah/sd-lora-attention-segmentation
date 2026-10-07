#!/bin/bash
#SBATCH --job-name=genlora
#SBATCH --partition=gpu_lowpriority
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=logs/narmyn_genlora_%x_%j.log
#
# One file for every rank AND every node. RANK and CONDA_ENV come from the
# environment; partition/gres/nodelist are just defaults you override on the
# sbatch line, so there is no need for a per-rank or per-node copy.
#
# The defaults above target renyi's spare a100 via gpu_lowpriority
# (PriorityTier=1): any gpu-partition job (PriorityTier=2) preempts this one,
# and PreemptMode=REQUEUE returns it to the queue rather than killing it.
# 06_generate_lora.py skips finished images, so a preemption costs only the
# image in flight.
#
# renyi (a100, preemptible, env trdp = torch 2.11+cu130):
#   sbatch --job-name=genlora_r8 --export=ALL,RANK=8 jobs/narmyn/generate_lora.slurm
#
# erdos / neumann (v100, NOT preemptible, env gen = torch 2.4.1+cu121):
#   their drivers (560.35.05 / 545.23.08) are too old for cu130, which needs
#   >= 580; cu121 needs only >= 525, hence the separate env.
#   sbatch --job-name=genlora_r8_erdos -p gpu -w erdos --gres=gpu:v100:1 \
#          --time=12:00:00 --export=ALL,RANK=8,CONDA_ENV=gen \
#          jobs/narmyn/generate_lora.slurm --start 12500
#
# Extra flags pass through: --start/--end to give each worker a disjoint slice
# (always do this when several run at once), or --checkpoint 3000 to sample an
# intermediate checkpoint instead of the final weights.

set -eo pipefail

RANK="${RANK:-8}"
CONDA_ENV="${CONDA_ENV:-trdp}"

source ~/.bashrc
conda activate "${CONDA_ENV}"

REPO="${REPO:-$HOME/sd-lora-attention-segmentation}"
# Data/outputs tree (cropped images, captions, weights, generations).
export TRDP_ROOT="${TRDP_ROOT:-$HOME/TRDP}"
cd "$REPO"
mkdir -p logs "$TRDP_ROOT/output/logs" "$TRDP_ROOT/output/lora_images/rank${RANK}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

echo "=== rank=${RANK} env=${CONDA_ENV} node=$(hostname -s) job=${SLURM_JOB_NAME} id=${SLURM_JOB_ID} restarts=${SLURM_RESTART_COUNT:-0} ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

python scripts/narmyn/06_generate_lora.py --rank "${RANK}" "$@"
