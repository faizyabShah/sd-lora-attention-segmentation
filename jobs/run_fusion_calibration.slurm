#!/bin/bash
#SBATCH --job-name=fusion_cal
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/fusion_cal_%A_%a.log
#
# Fusion and calibration package (phase 3 registry Section 1.2), one VOC-Sim
# class per array task. Resumable. Run from the repository root:
#
#   sbatch jobs/run_fusion_calibration.slurm
#   # after all tasks:
#   python scripts/select_propagation_cv.py \
#     --package-dir outputs/segmentation/controlled_attention_cv/fusion_calibration \
#     --spec configs/controlled_fusion_calibration_screens.json
#
# CONDA_ENV defaults to trdp; CRF_SITE (required) points at a built SimpleCRF denseCRF.
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p logs
echo "node=$(hostname -s) env=${CONDA_ENV:-trdp} shard=${SLURM_ARRAY_TASK_ID}"
python scripts/run_fusion_calibration.py --class-shard "${SLURM_ARRAY_TASK_ID}" \
  --crf-site "${CRF_SITE:?set CRF_SITE to a SimpleCRF denseCRF build for this Python}" "$@"
