#!/bin/bash
#SBATCH --job-name=sd15-metrics
#SBATCH --output=eval_%A_%a.out
#SBATCH --error=eval_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-24

# Submit this array with: sbatch run_evaluate.slurm
# After it completes, submit the CPU summary job with a dependency:
# sbatch --dependency=afterok:<array-job-id> run_evaluate_summary.slurm

source ~/.bashrc
conda activate trdp
set -euo pipefail

RANKS=(8 16 32 64 128)
SEEDS=(1 2 3 4 5)
RANK=${RANKS[$((SLURM_ARRAY_TASK_ID / 5))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID % 5))]}

python cityscapes_workflow.py evaluate --ranks "$RANK" --seeds "$SEED"
