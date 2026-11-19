#!/bin/bash
# CPU scoring (raw + DenseCRF R2). Usage:
#   sbatch --array=0-2 comparisons/official/score_cpu.slurm n        # N methods, 3 shards
#   sbatch comparisons/official/score_cpu.slurm brace                # BRACE counts (after the capture)
#SBATCH --job-name=n_official_score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/sidna/TRDP/external/n_side/logs/%x_%A_%a.out
#SBATCH --error=/home/sidna/TRDP/external/n_side/logs/%x_%A_%a.err
set -eo pipefail
cd /home/sidna/TRDP/sd-lora-attention-segmentation
export PATH=/home/sidna/miniconda3/envs/gen/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TMPDIR=/home/sidna/tmp_pip
export PYTHONPATH=/home/sidna/TRDP/external/n_side/deps/simplecrf_py311
SH=${SLURM_ARRAY_TASK_ID:-0}; NS=${SLURM_ARRAY_TASK_COUNT:-1}
if [ "$1" = "n" ]; then
  python comparisons/official/score_n.py --workers 40 --shard "$SH" --num-shards "$NS"
else
  python comparisons/official/brace_repro.py score --workers 40 --shard "$SH" --num-shards "$NS"
fi
