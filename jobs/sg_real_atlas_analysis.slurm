#!/bin/bash
# SG-real-atlas CPU analyses (Section 137). Usage: sbatch jobs/sg_real_atlas_analysis.slurm SCRIPT ARGS...
#SBATCH --job-name=sg-real-atlas-ana
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
cd /home/shafa/sd-lora-attention-segmentation
source ~/.bashrc
conda activate trdp
set -u
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
python "$@"
