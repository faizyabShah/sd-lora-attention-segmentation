#!/bin/bash
#SBATCH --job-name=n_frows_sclip
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/sidna/TRDP/external/n_side/official_f_rows/logs/%x_%A_%a.out
#SBATCH --error=/home/sidna/TRDP/external/n_side/official_f_rows/logs/%x_%A_%a.err
set -eo pipefail
cd /home/sidna/TRDP/sd-lora-attention-segmentation
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
/home/sidna/miniconda3/envs/goca/bin/python comparisons/official_f_rows/sclip_f.py ${SLURM_ARRAY_TASK_ID:-0} ${SLURM_ARRAY_TASK_COUNT:-1}
