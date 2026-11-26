#!/bin/bash
# runs/25 capture, side N. Usage: sbatch --array=0-1 capture.slurm   (2 shards over both datasets; resumable)
#SBATCH --job-name=n_ladder_nr
#SBATCH --partition=gpu
#SBATCH --nodelist=erdos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/home/sidna/TRDP/external/n_side/logs/%x_%A_%a.out
#SBATCH --error=/home/sidna/TRDP/external/n_side/logs/%x_%A_%a.err
set -eo pipefail
cd /home/sidna/TRDP/sd-lora-attention-segmentation
export PATH=/home/sidna/miniconda3/envs/gen/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TMPDIR=/home/sidna/tmp_pip PYTHONPATH=/home/sidna/TRDP/external/n_side/deps/simplecrf_py311
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch, diffusers, transformers; print(torch.__version__, diffusers.__version__, transformers.__version__)"
for ds in coco voc; do
  python comparisons/token_ladder_nr/run_nr.py capture --dataset "$ds" --shard "$SLURM_ARRAY_TASK_ID" --num-shards 2
done
