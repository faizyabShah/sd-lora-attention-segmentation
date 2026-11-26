#!/bin/bash
# Table 1 ladder capture, side N. Usage: sbatch --array=0-1 [--nodelist=erdos] capture.slurm [extra args]
#SBATCH --job-name=n_t1_ladder
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/sidna/TRDP/external/n_side/logs/%x_%A_%a.out
#SBATCH --error=/home/sidna/TRDP/external/n_side/logs/%x_%A_%a.err
set -eo pipefail
cd /home/sidna/TRDP/sd-lora-attention-segmentation
export PATH=/home/sidna/miniconda3/envs/gen/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
export TMPDIR=/home/sidna/tmp_pip PYTHONPATH=/home/sidna/TRDP/external/n_side/deps/simplecrf_py311
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
NS=${NUM_SHARDS:-2}
for ds in ${DATASETS:-coco voc}; do
  python comparisons/drafts/table1_ladder/t1_ladder.py capture --dataset "$ds" --shard "${SLURM_ARRAY_TASK_ID:-0}" --num-shards "$NS" "$@"
done
