#!/bin/bash
#SBATCH --job-name=attention-summary
#SBATCH --output=logs/attention_summary_%j.out
#SBATCH --error=logs/attention_summary_%j.err
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:30:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation
python scripts/summarize_stage1_stage2.py
