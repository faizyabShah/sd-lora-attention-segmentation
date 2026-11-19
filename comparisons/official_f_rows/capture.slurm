#!/bin/bash
# JC #3 remaining rows: OVAM / OVAM token / DAAM / SCLIP from one generation per item (wald V100). Resumable.
# Usage: sbatch [--export=ALL,ARGS="--dataset voc --shard 0 --num-shards 2"] capture.slurm
#SBATCH --job-name=n_frows_cap
#SBATCH --partition=gpu
#SBATCH --exclude=renyi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/home/sidna/TRDP/external/n_side/official_f_rows/logs/%x_%j.out
#SBATCH --error=/home/sidna/TRDP/external/n_side/official_f_rows/logs/%x_%j.err
set -eo pipefail
cd /home/sidna/TRDP/sd-lora-attention-segmentation
export PATH=/home/sidna/miniconda3/envs/gen/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TMPDIR=/home/sidna/tmp_pip PYTHONPATH=/home/sidna/TRDP/external/n_side/official_f_rows/deps
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch, diffusers, transformers; print(torch.__version__, diffusers.__version__, transformers.__version__)"
python comparisons/official_f_rows/capture_f.py $ARGS
