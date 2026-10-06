#!/bin/bash
#SBATCH --job-name=sd15-all-metrics
#SBATCH --output=logs/sd15_all_metrics_%j.out
#SBATCH --error=logs/sd15_all_metrics_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# One job, sequential runs: frozen seeds 1-4, then all rank/seed LoRAs.
# Re-submit the same script if the time limit is reached; completed runs are skipped.
source ~/.bashrc
conda activate trdp
set -euo pipefail
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR"
python cityscapes_workflow.py evaluate --frozen-seeds 1 2 3 4 --skip-existing
