#!/bin/bash
# Usage: sbatch -c <cpus> fast_score.slurm <voc|coco> <ladder|t1> <shard> <num_shards>
#SBATCH --job-name=n_fast_score
#SBATCH --partition=cpu,gpu_lowpriority
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/home/sidna/TRDP/external/n_side/logs/%x_%j.out
#SBATCH --error=/home/sidna/TRDP/external/n_side/logs/%x_%j.err
set -eo pipefail
cd /home/sidna/TRDP/sd-lora-attention-segmentation
export PATH=/home/sidna/miniconda3/envs/gen/bin:$PATH OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/home/sidna/TRDP/external/n_side/deps/simplecrf_py311
python comparisons/drafts/fast_score.py "$1" "$2" --shard "$3" --num-shards "$4" --workers "$SLURM_CPUS_PER_TASK"
