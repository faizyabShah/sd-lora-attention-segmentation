#!/bin/bash
#SBATCH --job-name=sd15-metrics-summary
#SBATCH --output=eval_summary_%j.out
#SBATCH --error=eval_summary_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
python cityscapes_workflow.py summarize
