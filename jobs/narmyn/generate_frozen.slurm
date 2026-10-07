#!/bin/bash
#SBATCH --job-name=genfrozen
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/narmyn_generate_frozen_%j.log
# renyi's a100 is roughly 2x erdos's v100 for SD 1.5 inference, and this job
# needs every bit of it. To use erdos instead, swap the two lines above for:
#   #SBATCH --nodelist=erdos
#   #SBATCH --gres=gpu:v100:1
#
# 24,995 images at 768x768 will NOT fit in one 12h window. Resume-skip makes
# that safe: just resubmit and finished images are skipped. To halve wall time,
# run two jobs over disjoint halves on renyi's two a100s:
#   sbatch jobs/narmyn/generate_frozen.slurm --start 0     --end 12498
#   sbatch jobs/narmyn/generate_frozen.slurm --start 12498
# stderr is merged into --output. Submit from ~/TRDP so the log path resolves.

set -eo pipefail

source ~/.bashrc
conda activate trdp

REPO="${REPO:-$HOME/sd-lora-attention-segmentation}"
# Data/outputs tree (cropped images, captions, weights, generations).
export TRDP_ROOT="${TRDP_ROOT:-$HOME/TRDP}"
cd "$REPO"
mkdir -p logs "$TRDP_ROOT/output/logs" output/frozen_images

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

python scripts/narmyn/03_generate_frozen.py "$@"
