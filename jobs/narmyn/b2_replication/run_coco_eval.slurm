#!/bin/bash
#SBATCH --job-name=b2rep_coco
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --exclude=renyi,wald
#SBATCH --array=0-19
#
# Independent replication of Codex's B2a/S4/B2b token package (protocol Section 65):
# Codex's committed scripts and plan, Claude's environment (gen, V100), separate output.
# Run from the repository root via jobs/narmyn/b2_replication/submit.sh.
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-gen}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
python scripts/narmyn/attention/b2_replication_eval.py --dataset coco --class-shard "${SLURM_ARRAY_TASK_ID}" --learned-dir outputs/segmentation/controlled_attention_cv/token_learning_b2_replication --output-dir outputs/segmentation/controlled_attention_cv/token_learning_b2_replication --crf-site "${CRF_SITE:?set CRF_SITE to a SimpleCRF denseCRF build for this Python}"
