#!/bin/bash
#SBATCH --job-name=coco_token
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/coco_token_%A_%a.log
#
# Few-shot COCO-Cap transfer (OVAM VOC tokens; protocol Section 55), one
# class per array task. Resumable. Nothing is chosen on COCO-Cap. Run from the
# repository root:
#
#   sbatch jobs/run_coco_cap_token_transfer.slurm
#   sbatch jobs/run_coco_cap_token_transfer.slurm --self-test-voc --max-samples 2   # path check on VOC-Sim
#
# CONDA_ENV defaults to trdp (renyi only); CRF_SITE (required) points at a built SimpleCRF denseCRF.
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p logs
echo "node=$(hostname -s) env=${CONDA_ENV:-trdp} shard=${SLURM_ARRAY_TASK_ID}"
python scripts/run_coco_cap_token_transfer.py --class-shard "${SLURM_ARRAY_TASK_ID}" \
  --crf-site "${CRF_SITE:?set CRF_SITE to a SimpleCRF denseCRF build for this Python}" "$@"
