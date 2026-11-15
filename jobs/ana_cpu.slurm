#!/bin/bash
# Map analysis (CPU). Usage: sbatch jobs/ana_cpu.slurm scripts/<script>.py <args>
#SBATCH --job-name=ana-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -eo pipefail
cd /home/shafa/sd-lora-attention-segmentation
source ~/.bashrc
conda activate trdp
set -u
export BRACE_CACHE_DIR=/home/shafa/sg_comp PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1
python "$@"
