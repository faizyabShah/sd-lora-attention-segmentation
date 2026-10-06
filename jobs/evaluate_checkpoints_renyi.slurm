#!/bin/bash
#SBATCH --job-name=eval-sd15-checkpoints
#SBATCH --output=logs/eval_sd15_checkpoints_%j.out
#SBATCH --error=logs/eval_sd15_checkpoints_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Submitted with an afterok dependency on both checkpoint-generation jobs.
# Sequential and resumable: completed per-run metric files are skipped.
source ~/.bashrc
conda activate trdp
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"

python cityscapes_workflow.py evaluate-checkpoints \
    --checkpoints 1200 2400 3600 \
    --skip-existing
