#!/bin/bash
#SBATCH --job-name=repair-resume-metrics
#SBATCH --output=logs/repair_resume_metrics_%j.out
#SBATCH --error=logs/repair_resume_metrics_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Repairs missing/empty rank-32 seed-5 images, then resumes unfinished metrics.
source ~/.bashrc
conda activate trdp
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
python cityscapes_workflow.py generate --rank 32 --seed 5
python cityscapes_workflow.py evaluate --frozen-seeds 1 2 3 4 --skip-existing
