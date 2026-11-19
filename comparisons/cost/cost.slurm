#!/bin/bash
# JC #8 cost measurement. Usage: sbatch --nodelist=<node> cost.slurm <n_per_set> <out.json>
#SBATCH --job-name=n_cost
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=/home/sidna/TRDP/external/n_side/logs/%x_%j.out
#SBATCH --error=/home/sidna/TRDP/external/n_side/logs/%x_%j.err
set -eo pipefail
cd /home/sidna/TRDP/sd-lora-attention-segmentation
export PATH=/home/sidna/miniconda3/envs/gen/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TMPDIR=/home/sidna/tmp_pip PYTHONPATH=/home/sidna/TRDP/external/n_side/deps/simplecrf_py311
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
lscpu | grep "Model name"
python comparisons/cost/measure_cost.py --n "$1" --out "$2"
