#!/bin/bash
#SBATCH --job-name=merge-propagation
#SBATCH --output=logs/merge_propagation_%j.out
#SBATCH --error=logs/merge_propagation_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation
mkdir -p logs
python scripts/select_propagation_cv.py \
  --package-dir outputs/segmentation/controlled_attention_cv/propagation_package \
  --spec configs/controlled_propagation_screens.json
