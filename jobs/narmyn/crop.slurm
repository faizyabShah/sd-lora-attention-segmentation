#!/bin/bash
#SBATCH --job-name=crop
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/narmyn_crop_%j.log
# CPU-only preprocessing -- no --gres; a GPU does nothing for PNG encoding.
# 32 cores matches hajos/kemeny's physical core count (they report 64 CPUs with
# ThreadsPerCore=2). euclid has only 24, so SLURM may place us there instead --
# 01_crop.py sizes its pool from $SLURM_CPUS_PER_TASK either way.
# stderr is merged into --output. Submit from ~/TRDP so the log path resolves.

set -eo pipefail

source ~/.bashrc
conda activate trdp

REPO="${REPO:-$HOME/sd-lora-attention-segmentation}"
# Data/outputs tree (cropped images, captions, weights, generations).
export TRDP_ROOT="${TRDP_ROOT:-$HOME/TRDP}"
cd "$REPO"
mkdir -p logs "$TRDP_ROOT/output/logs"

python scripts/narmyn/01_crop.py
