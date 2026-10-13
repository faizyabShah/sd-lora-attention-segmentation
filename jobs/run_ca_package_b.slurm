#!/bin/bash
#SBATCH --job-name=ca_pkg_b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/ca_pkg_b_%A_%a.log
#
# CA package B (CA-L, CA-H, CA-A) of the controlled protocol, one VOC-Sim class
# per array task. Resumable. Run from the repository root:
#
#   sbatch jobs/run_ca_package_b.slurm
#
# CONDA_ENV defaults to trdp; set it to an environment whose torch build runs on
# the node (e.g. a cu121 build on V100 nodes).
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p logs
echo "node=$(hostname -s) env=${CONDA_ENV:-trdp} shard=${SLURM_ARRAY_TASK_ID}"
python scripts/run_ca_package_b.py --class-shard "${SLURM_ARRAY_TASK_ID}" "$@"
