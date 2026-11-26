#!/bin/bash
# CPU scoring of the JC #3 remaining rows. Usage: sbatch [--array=0-N] score.slurm raw|crf [--taus f] [--sets ...]
#SBATCH --job-name=n_frows_score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=40
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/home/sidna/TRDP/external/n_side/official_f_rows/logs/%x_%A_%a.out
#SBATCH --error=/home/sidna/TRDP/external/n_side/official_f_rows/logs/%x_%A_%a.err
set -eo pipefail
cd /home/sidna/TRDP/sd-lora-attention-segmentation
export PATH=/home/sidna/miniconda3/envs/gen/bin:$PATH
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TMPDIR=/home/sidna/tmp_pip
export PYTHONPATH=/home/sidna/TRDP/external/n_side/deps/simplecrf_py311
SH=${SLURM_ARRAY_TASK_ID:-0}; NS=${SLURM_ARRAY_TASK_COUNT:-1}
python comparisons/official_f_rows/score_f.py "$@" --workers ${SLURM_CPUS_PER_TASK:-40} --shard "$SH" --num-shards "$NS"
