#!/bin/bash
#SBATCH --job-name=ca_confirm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/ca_confirm_%A_%a.log
#
# CA-C combined confirmation (protocol Section 25), one VOC-Sim class
# per array task. Resumable. Run from the repository root:
#
#   sbatch jobs/run_ca_confirmation.slurm
#
# CONDA_ENV defaults to trdp; set it to an environment whose torch build runs on
# the node (e.g. a cu121 build on V100 nodes).
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p logs
echo "node=$(hostname -s) env=${CONDA_ENV:-trdp} shard=${SLURM_ARRAY_TASK_ID}"
python scripts/run_ca_confirmation.py --class-shard "${SLURM_ARRAY_TASK_ID}" "$@"
