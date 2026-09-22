#!/bin/bash
#SBATCH --job-name=sd15-frozen-seeds
#SBATCH --output=frozen_%A_%a.out
#SBATCH --error=frozen_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7:00:00
#SBATCH --array=1-5

source ~/.bashrc
conda activate trdp
set -euo pipefail

python cityscapes_workflow.py generate --frozen --seed "$SLURM_ARRAY_TASK_ID"
