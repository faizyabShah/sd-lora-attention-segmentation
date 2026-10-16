#!/bin/bash
#SBATCH --job-name=merge-p3c
#SBATCH --output=logs/merge_p3c_%j.out
#SBATCH --error=logs/merge_p3c_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
source ~/.bashrc
conda activate trdp
set -euo pipefail
cd /home/shafa/sd-lora-attention-segmentation
mkdir -p logs
python scripts/select_propagation_cv.py \
  --package-dir outputs/segmentation/controlled_attention_cv/phase3_confirmation \
  --spec configs/controlled_phase3_confirmation.json
python scripts/finalize_phase3_confirmation.py \
  --package-dir outputs/segmentation/controlled_attention_cv/phase3_confirmation
