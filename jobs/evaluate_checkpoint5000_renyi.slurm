#!/bin/bash
#SBATCH --job-name=eval-sd15-step5000
#SBATCH --output=logs/eval_sd15_step5000_%j.out
#SBATCH --error=logs/eval_sd15_step5000_%j.err
#SBATCH --partition=gpu
#SBATCH --nodelist=renyi
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Evaluate final step-5000 LoRAs on the exact same 1,000 caption indices used
# for checkpoints 1200, 2400, and 3600. Existing frozen metrics are skipped.
source ~/.bashrc
conda activate trdp
set -euo pipefail
export TOKENIZERS_PARALLELISM=false
cd "$SLURM_SUBMIT_DIR"

python cityscapes_workflow.py prepare-final-subsets
python cityscapes_workflow.py evaluate-checkpoints \
    --checkpoints 5000 \
    --skip-existing
