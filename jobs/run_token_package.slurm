#!/bin/bash
#SBATCH --job-name=tokens
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/tokens_%A_%a.log
#
# Token track B1/B1b (phase 3 registry Section 2), one VOC-Sim
# class per array task. Resumable. Run from the repository root:
#
#   sbatch jobs/run_token_package.slurm
#   # after all tasks:
#   python scripts/select_propagation_cv.py \
#     --package-dir outputs/segmentation/controlled_attention_cv/tokens \
#     --spec configs/controlled_token_screens.json
#
# CONDA_ENV defaults to trdp; CRF_SITE points at a built SimpleCRF denseCRF.
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p logs
echo "node=$(hostname -s) env=${CONDA_ENV:-trdp} shard=${SLURM_ARRAY_TASK_ID}"
python scripts/run_token_package.py --class-shard "${SLURM_ARRAY_TASK_ID}" \
  --crf-site "${CRF_SITE:-$HOME/TRDP/output/ovam_token_pilot/crf_site}" "$@"
