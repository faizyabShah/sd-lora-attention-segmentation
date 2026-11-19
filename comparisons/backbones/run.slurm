#!/bin/bash
# JC #5 single-pass capture. Usage: sbatch run.slurm <subcommand and args for single_pass.py>
#SBATCH --job-name=n-bb
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/sidna/TRDP/external/n_side/backbones/logs/%x_%j.out
#SBATCH --error=/home/sidna/TRDP/external/n_side/backbones/logs/%x_%j.err
set -eo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8 PYTHONWARNINGS=ignore
cd /home/sidna/TRDP/sd-lora-attention-segmentation
hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
date; /home/sidna/miniconda3/envs/gen/bin/python comparisons/backbones/single_pass.py "$@"; date
