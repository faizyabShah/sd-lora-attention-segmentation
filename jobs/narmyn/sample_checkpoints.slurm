#!/bin/bash
#SBATCH --job-name=ckptgrid
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/narmyn_ckptgrid_%x_%j.log
#
# Only 1 GPU and ~1 minute of work per rank, so this slots into renyi's spare
# card without disturbing a 2-GPU training job.
#
#   sbatch --job-name=ckptgrid_r8 --export=ALL,RANK=8 jobs/narmyn/sample_checkpoints.slurm
#
# Extra flags pass through, e.g. --every 2 to halve the number of rows.

set -eo pipefail

RANK="${RANK:-8}"
# erdos/neumann drivers cannot run trdp's cu130 torch; pass CONDA_ENV=gen there.
CONDA_ENV="${CONDA_ENV:-trdp}"

source ~/.bashrc
conda activate "${CONDA_ENV}"

REPO="${REPO:-$HOME/sd-lora-attention-segmentation}"
# Data/outputs tree (cropped images, captions, weights, generations).
export TRDP_ROOT="${TRDP_ROOT:-$HOME/TRDP}"
cd "$REPO"
mkdir -p logs "$TRDP_ROOT/output/logs"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

python scripts/narmyn/05_sample_checkpoints.py --rank "${RANK}" "$@"
