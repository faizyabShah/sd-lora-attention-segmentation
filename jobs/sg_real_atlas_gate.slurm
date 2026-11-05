#!/bin/bash
# SG-real-atlas gate: replay/storage check on the smoke capture; non-zero exit blocks the full run.
#SBATCH --job-name=sg-real-atlas-gate
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
cd /home/shafa/sd-lora-attention-segmentation
source ~/.bashrc
conda activate trdp
set -u
python scripts/check_sg_real_atlas_replay.py /home/shafa/sg_real_atlas_smoke_v100 --gate
