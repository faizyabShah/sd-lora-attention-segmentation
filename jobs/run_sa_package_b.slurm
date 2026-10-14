#!/bin/bash
#SBATCH --job-name=sa_pkg_b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --array=0-19
#SBATCH --output=logs/sa_pkg_b_%A_%a.log
#
# SA package B (SA-L, SA-H, A^T x check) of the controlled protocol, one VOC-Sim
# class per array task. Resumable. Run from the repository root once the CA
# configuration is frozen (protocol Section 11) and written to a seed spec with
# "frozen": true:
#
#   sbatch jobs/run_sa_package_b.slurm --ca-seed configs/controlled_sa_seed_frozen.json
#   # after all tasks: python scripts/merge_sa_package_b.py
#
# CONDA_ENV defaults to trdp; set it to an environment whose torch build runs on
# the node (e.g. gen on the V100 nodes). Keep one GPU type for a whole package.
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-trdp}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
mkdir -p logs
echo "node=$(hostname -s) env=${CONDA_ENV:-trdp} shard=${SLURM_ARRAY_TASK_ID}"
python scripts/run_sa_package_b.py --class-shard "${SLURM_ARRAY_TASK_ID}" "$@"
