#!/bin/bash
#SBATCH --job-name=b2rep_b2a
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --exclude=renyi,wald
#SBATCH --array=0-58
#
# Independent replication of Codex's B2a/S4/B2b token package (protocol Section 65):
# Codex's committed scripts and plan, Claude's environment (gen, V100), separate output.
# Run from the repository root via jobs/narmyn/b2_replication/submit.sh.
set -eo pipefail
source ~/.bashrc
conda activate "${CONDA_ENV:-gen}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1
python scripts/optimize_b2_tokens.py --track b2a --task-id "${SLURM_ARRAY_TASK_ID}" --output-dir outputs/segmentation/controlled_attention_cv/token_learning_b2_replication
