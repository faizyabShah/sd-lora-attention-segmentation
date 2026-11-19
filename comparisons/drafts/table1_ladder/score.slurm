#!/bin/bash
# Table 1 ladder scoring (CPU, resumable: skips scored images). Usage: sbatch [-c N] score.slurm
#SBATCH --job-name=n_t1_score
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --output=/home/sidna/TRDP/external/n_side/logs/%x_%j.out
#SBATCH --error=/home/sidna/TRDP/external/n_side/logs/%x_%j.err
set -eo pipefail
cd /home/sidna/TRDP/sd-lora-attention-segmentation
export PATH=/home/sidna/miniconda3/envs/gen/bin:$PATH OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export PYTHONPATH=/home/sidna/TRDP/external/n_side/deps/simplecrf_py311
python comparisons/drafts/table1_ladder/t1_ladder.py score --workers "$SLURM_CPUS_PER_TASK" ${SETS:+--sets $SETS}
