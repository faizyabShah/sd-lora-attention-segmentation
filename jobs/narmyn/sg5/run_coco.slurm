#!/bin/bash
#SBATCH --job-name=sg5_coco
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclude=renyi,wald
#SBATCH --array=0-19
#
# SG5 contrastive CA seed (protocol Section 66), one COCO-Cap class per task. Resumable.
# One environment for the whole package (R7): gen, V100. Run via jobs/narmyn/sg5/submit.sh.
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-gen}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
echo "node=$(hostname -s) env=${CONDA_ENV:-gen} shard=${SLURM_ARRAY_TASK_ID}"
python scripts/narmyn/attention/run_sg5_contrastive_seed.py --dataset coco --class-shard "${SLURM_ARRAY_TASK_ID}" --crf-site "${CRF_SITE:?set CRF_SITE to a SimpleCRF denseCRF build for this Python}"
