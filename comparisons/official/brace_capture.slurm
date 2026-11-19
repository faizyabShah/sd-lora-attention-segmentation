#!/bin/bash
# BRACE (CAg x V3) per-image reproduction of runs/20 on one V100. Resumable (skips finished images).
#SBATCH --job-name=n_brace_repro
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --exclude=erdos,renyi
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/home/sidna/TRDP/external/n_side/logs/%x_%j.out
#SBATCH --error=/home/sidna/TRDP/external/n_side/logs/%x_%j.err
set -eo pipefail
cd /home/sidna/TRDP/sd-lora-attention-segmentation
export PATH=/home/sidna/miniconda3/envs/gen/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TMPDIR=/home/sidna/tmp_pip BRACE_CACHE_DIR=/home/sidna/TRDP/external/n_side/brace_repro/cache
export PYTHONPATH=/home/sidna/TRDP/external/n_side/deps/simplecrf_py311
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch, diffusers, transformers; print(torch.__version__, diffusers.__version__, transformers.__version__)"
for ds in ${DATASETS:-voc coco}; do
  python comparisons/official/brace_repro.py capture --dataset "$ds"
done
