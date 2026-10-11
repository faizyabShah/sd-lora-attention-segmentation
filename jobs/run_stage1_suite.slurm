#!/bin/bash
#SBATCH --job-name=attention-stage1
#SBATCH --output=logs/attention_stage1_%j.out
#SBATCH --error=logs/attention_stage1_%j.err
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation

python scripts/run_stage1_suite.py \
    --capture-dir outputs/attention/voc_sim \
    --output-dir outputs/segmentation/voc_sim_100_stage1_suite
